local cjson = require 'cjson'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'

local eval_utils = {}

function eval_utils.eval_split(kwargs)
  local model = utils.getopt(kwargs, 'model')
  local loader = utils.getopt(kwargs, 'loader')
  local split = utils.getopt(kwargs, 'split', 'val')
  local max_images = utils.getopt(kwargs, 'max_images', -1)
  local id = utils.getopt(kwargs, 'id', '')
  local dtype = utils.getopt(kwargs, 'dtype', 'torch.FloatTensor')
  assert(split == 'val' or split == 'test', 'split must be "val" or "test"')
  local split_to_int = {val=1, test=2}
  split = split_to_int[split]
  print('using split ', split)
  
  model:evaluate()
  loader:resetIterator(split)
  local evaluator = DenseCaptioningEvaluator{id=id}
   
  local counter = 0
  local num_box = 0
  local num_caption = 0
  
  while true do
    counter = counter + 1
    
    local data = {}
    local loader_kwargs = {split=split, iterate=true}
    local img, gt_boxes, gt_labels, info, _ = loader:getBatch(loader_kwargs)
    local data = {
      image = img:type(dtype),
      gt_boxes = gt_boxes:type(dtype),
      gt_labels = gt_labels:type(dtype),
    }
    info = info[1]
    
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false

    local boxes, logprobs, captions = model:forward_test(data.image)    
    
    num_box = num_box + boxes:size(1)
    num_caption = num_caption + #captions
    
    local gt_captions = model.nets.language_model:decodeSequence(gt_labels[1])
    evaluator:addResult(logprobs, boxes, captions, gt_boxes[1], gt_captions)
    
    local msg = 'Processed image %s (%d / %d) of split %d, detected %d regions'
    local num_images = info.split_bounds[2]
    if max_images > 0 then num_images = math.min(num_images, max_images) end
    local num_boxes = boxes:size(1)
    print(string.format(msg, info.filename, counter, num_images, split, num_boxes))

    if max_images > 0 and counter >= max_images then break end
    if info.split_bounds[1] == info.split_bounds[2] then break end
  end
  
  local results = evaluator:evaluate()
  print(string.format('Recall: %f', 100 * results.recall))
  print(string.format('METEOR: %f', 100 * results.meteor))
  print(string.format('num of box: %f', num_box/counter))
  print(string.format('num of captions: %f', num_caption/counter))
  
  local out = {
    results=results,
  }
  return out
end


function eval_utils.score_captions(records)
  utils.write_json('eval/input.json', records)
  os.execute('python eval/meteor_bridge.py')
  local blob = utils.read_json('eval/output.json')
  return blob
end


local function pluck_boxes(ix, boxes, text)

  local N = #ix
  local new_boxes = torch.zeros(N, 4)
  local new_text = {}

  for i=1,N do
    local ixi = ix[i]
    local n = ixi:nElement()
    local bsub = boxes:index(1, ixi)
    local newbox = torch.mean(bsub, 1)
    new_boxes[i] = newbox

    local texts = {}
    if text then
      for j=1,n do
        table.insert(texts, text[ixi[j]])
      end
    end
    table.insert(new_text, texts)
  end

  return new_boxes, new_text
end


local DenseCaptioningEvaluator = torch.class('DenseCaptioningEvaluator')
function DenseCaptioningEvaluator:__init(opt)
  self.all_logprobs = {}
  self.records = {}
  self.n = 1
  self.npos = 0
  self.id = utils.getopt(opt, 'id', '')
  self.captions = {}
end

function DenseCaptioningEvaluator:addResult(logprobs, boxes, text, target_boxes, target_text)
  assert(logprobs:size(1) == boxes:size(1))
  assert(boxes:nDimension() == 2)
  
  table.insert(self.captions, text)
  
  boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
  target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes)

  boxes = boxes:float()
  logprobs = logprobs[{ {}, 1 }]:double() 
  target_boxes = target_boxes:float()

  local Y,IX = torch.sort(logprobs,1,true)
  
  local nd = #text
  local nt = #target_text 
  
  for j=1,nt do
    local ovmax = 0
    local imax = -1
    local jmax = -1
    local reference = {}
    
    for d=1,nd do
      local ii = d
      table.insert(reference,text[ii])
    end
    local ok = 1

    local record = {}
    record.ok = ok 
    record.ov = 1
    record.candidate = target_text[j]

    record.references = reference
    if record.references == nil then record.references = {} end
    record.imgid = self.n
    table.insert(self.records, record)
  end

  self.n = self.n + 1
  self.npos = self.npos + nt
  table.insert(self.all_logprobs, Y:double()) 
end

function DenseCaptioningEvaluator:evaluate(verbose)
  if verbose == nil then verbose = true end
  local min_overlaps = {0}
  local min_scores = {-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25}

  local logprobs = torch.cat(self.all_logprobs, 1) 
  local blob = eval_utils.score_captions(self.records, self.id) 
  local scores = blob.scores 
  collectgarbage()
  collectgarbage()

  if verbose then
    for k=1,#self.records do
      local record = self.records[k]
      if record.ov > 0 and record.ok == 1 and k % 1000 == 0 then
        local txtgt = ''
        assert(type(record.references) == "table")
        for kk,vv in pairs(record.references) do txtgt = txtgt .. vv .. '. ' end
        print(string.format('IMG %d PRED: %s, GT: %s, OK: %d, OV: %f SCORE: %f',
              record.imgid, record.candidate, txtgt, record.ok, record.ov, scores[k]))
      end  
    end
  end

  local recall_results = {}
  local det_results = {}
  for foo, min_overlap in pairs(min_overlaps) do
    for foo2, min_score in pairs(min_scores) do

      local n = #scores
      local tp = torch.zeros(n)
      local tn = torch.zeros(n)
      for i=1,n do
        local ii = i
        local r = self.records[ii]

        if not r.references then
          tn[i] = 1 
        else
          local score = scores[ii]
          if r.ov >= min_overlap and r.ok == 1 and score > min_score then
            tp[i] = 1
          else
            tn[i] = 1
          end
        end
      end

      tn = torch.cumsum(tn,1)
      tp = torch.cumsum(tp,1)
      local rec = torch.div(tp, self.npos)
      local prec = torch.cdiv(tp, tn + tp)

      local recall = 0
      local recallN = 0
      for t=0,1,0.01 do
        local mask = torch.ge(rec, t):double()
        local prec_masked = torch.cmul(prec:double(), mask)
        local p = torch.max(prec_masked)
        recall = recall + p
        recallN = recallN + 1
      end
      recall = recall / recallN

      if min_score == -1 then
        det_results['ov' .. min_overlap] = recall
      else
        recall_results['ov' .. min_overlap .. '_score' .. min_score] = recall
      end
    end
  end

  local mrecall = utils.average_values(recall_results)
  local METEOR = blob.average_score
  
  local results = {recall = mrecall, meteor = METEOR}
  return results
end

function DenseCaptioningEvaluator:numAdded()
  return self.n - 1
end

return eval_utils
