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
  
  local ap_results = evaluator:evaluate()
  print(string.format('mAP: %f', 100 * ap_results.map))
  
  local out = {
    ap_results=ap_results,
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
end

function DenseCaptioningEvaluator:addResult(logprobs, boxes, text, target_boxes, target_text)
  assert(logprobs:size(1) == boxes:size(1))
  assert(logprobs:size(1)*(logprobs:size(1)-1) == #text)
  assert(target_boxes:size(1) == #target_text*2)
  assert(boxes:nDimension() == 2)

  boxes = box_utils.xcycwh_to_x1y1x2y2(boxes)
  target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes)

  boxes = boxes:float()
  logprobs = logprobs[{ {}, 1 }]:double()
  target_boxes = target_boxes:float()

  local Y,IX = torch.sort(logprobs,1,true)
  
  local nd = boxes:size(1)
  local nt = #target_text
  local used = torch.zeros(nt)
  
  
  for d1=1,nd do
    for d2=1,nd do
      if not(d1==d2) then
        local ii = d1
        local jj = d2
        
        local bb1 = boxes[ii]
        local bb2 = boxes[jj]

        local ovmax = 0
        local jmax = -1
        for j=1,nt do
          
          local bbgt1 = target_boxes[j*2-1]
          local bbgt2 = target_boxes[j*2]
          
          --------------------------------------------------------------------------
          local bi1 = {math.max(bb1[1],bbgt1[1]), math.max(bb1[2],bbgt1[2]),
                      math.min(bb1[3],bbgt1[3]), math.min(bb1[4],bbgt1[4])}
          local iw1 = bi1[3]-bi1[1]+1
          local ih1 = bi1[4]-bi1[2]+1
          ------------------------------------------------------------------------
          local bi2 = {math.max(bb2[1],bbgt2[1]), math.max(bb2[2],bbgt2[2]),
                      math.min(bb2[3],bbgt2[3]), math.min(bb2[4],bbgt2[4])}
          local iw2 = bi2[3]-bi2[1]+1
          local ih2 = bi2[4]-bi2[2]+1
          -------------------------------------------------------------------------
          
          if iw1>0 and ih1>0 and iw2>0 and ih2>0 then
            local ua1 = (bb1[3]-bb1[1]+1)*(bb1[4]-bb1[2]+1)+
                       (bbgt1[3]-bbgt1[1]+1)*(bbgt1[4]-bbgt1[2]+1)-iw1*ih1
            local ua2 = (bb2[3]-bb2[1]+1)*(bb2[4]-bb2[2]+1)+
                       (bbgt2[3]-bbgt2[1]+1)*(bbgt2[4]-bbgt2[2]+1)-iw2*ih2
            
            --local ov = (iw1*ih1/ua1 + iw2*ih2/ua2)/2 --- ave of intersection (mean)
            local ov = torch.min(torch.Tensor{iw1*ih1/ua1 ,iw2*ih2/ua2})----------both above threshold (and)
            --local ov = torch.max(torch.Tensor{iw1*ih1/ua1 ,iw2*ih2/ua2})----------both above threshold (or)
            
            if ov > ovmax then
              ovmax = ov
              jmax = j
            end
          end
        end

        local ok = 1
        if used[jmax] == 0 then
          used[jmax] = 1 
        end

        local record = {}
        record.ok = ok 
        record.ov = ovmax
        record.candidate = text[(ii-1)*(nd-1) +  jj]
        
        record.references = {target_text[jmax]} 
        if record.references == nil then record.references = {} end
        record.imgid = self.n
        table.insert(self.records, record)
      end
    end
  end
  
  self.n = self.n + 1
  self.npos = self.npos + nt
  table.insert(self.all_logprobs, Y:double()) 
end

function DenseCaptioningEvaluator:evaluate(verbose)
  if verbose == nil then verbose = true end
  local min_overlaps = {0.2, 0.3, 0.4, 0.5,0.6}
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

  local ap_results = {}
  local det_results = {}
  for foo, min_overlap in pairs(min_overlaps) do
    for foo2, min_score in pairs(min_scores) do

      local n = #scores
      local tp = torch.zeros(n)
      local fp = torch.zeros(n)
      for i=1,n do
        local ii = i
        local r = self.records[ii]

        if not r.references then
          fp[i] = 1
        else
          local score = scores[ii]
          if r.ov >= min_overlap and r.ok == 1 and score > min_score then
            tp[i] = 1
          else
            fp[i] = 1
          end
        end
      end

      fp = torch.cumsum(fp,1)
      tp = torch.cumsum(tp,1)
      local rec = torch.div(tp, self.npos)
      local prec = torch.cdiv(tp, fp + tp)

      local ap = 0
      local apn = 0
      for t=0,1,0.01 do
        local mask = torch.ge(rec, t):double()
        local prec_masked = torch.cmul(prec:double(), mask)
        local p = torch.max(prec_masked)
        ap = ap + p
        apn = apn + 1
      end
      ap = ap / apn

      if min_score == -1 then
        det_results['ov' .. min_overlap] = ap
      else
        ap_results['ov' .. min_overlap .. '_score' .. min_score] = ap
      end
    end
  end

  local map = utils.average_values(ap_results)
  local detmap = utils.average_values(det_results)

  local results = {map = map, ap_breakdown = ap_results, detmap = detmap, det_breakdown = det_results}
  return results
end

function DenseCaptioningEvaluator:numAdded()
  return self.n - 1
end

return eval_utils
