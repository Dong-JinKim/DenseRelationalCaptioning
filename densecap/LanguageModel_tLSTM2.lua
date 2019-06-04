require 'nn'
require 'torch-rnn'
require 'nngraph'

local utils = require 'densecap.utils'

local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')

function LM:__init(opt)
  parent.__init(self)
end

function LM:decodeSequence(seq)
  local delimiter = ' '
  local captions = {}
  local N, T = seq:size(1), seq:size(2)
  for i = 1, N do
    local caption = ''
    for t = 1, T do
      local idx = seq[{i, t}]
      if idx == self.END_TOKEN or idx == 0 then break end
      if t > 1 then
        caption = caption .. delimiter
      end
      caption = caption .. self.idx_to_token[idx]
    end
    table.insert(captions, caption)
  end
  return captions
end

function LM:updateOutput(input)
  self.recompute_backward = true
  local union_vectors = input[1]
  local gt_sequence = input[2]
  local spatial = input[3]:cuda() 
  
  if ##union_vectors == 0  then
    self._forward_sampled = false
    return 0
  end
  
  local subj_vectors = input[4][1]
  local obj_vectors = input[4][2]

  self._forward_sampled = true
  return self:sample(union_vectors, subj_vectors, obj_vectors, spatial)
end

function LM:getTarget(gt_sequence)
  local gt_sequence_long = gt_sequence:long()
  local N, T = gt_sequence:size(1), gt_sequence:size(2)
  local target = torch.LongTensor(N, T + 2):zero()
  target[{{}, {2, T + 1}}]:copy(gt_sequence)
  for i = 1, N do
    for t = 2, T + 2 do
      if target[{i, t}] == 0 then
        target[{i, t}] = self.END_TOKEN
        break
      end
    end
  end
  return target:type(gt_sequence:type())
end


function LM:sample(union_vectors, subj_vectors, obj_vectors,spatial)
  local N, T = union_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(union_vectors:type())
  
  for i = 1, #self.rnn1 do
    local layer = self.rnn1:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end
  for i = 1, #self.rnn2 do
    local layer = self.rnn2:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end
  for i = 1, #self.rnn3 do
    local layer = self.rnn3:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)
  self.view_out2:resetSize(N, 1, -1)

  local union_vecs_encoded
  local subj_vecs_encoded
  local obj_vecs_encoded
  
  union_vecs_encoded  = self.image_encoder_real:forward{union_vectors,spatial}
  subj_vecs_encoded  = self.image_encoder2:forward(subj_vectors)
  obj_vecs_encoded  = self.image_encoder3:forward(obj_vectors)
  
  self.rnn1:forward(union_vecs_encoded)
  self.rnn2:forward(subj_vecs_encoded)
  self.rnn3:forward(obj_vecs_encoded)
  
  local GO_STOP=torch.LongTensor(N):fill(1)

  for t = 1, T do
    local words = nil
    if t == 1 then
      words = torch.LongTensor(N, 1):fill(self.START_TOKEN)
    else
      words = seq[{{}, {t-1, t-1}}]
    end
    local wordvecs = self.lookup_table:forward(words)
    local wordvecs2 = self.lookup_table2:forward(words)
    local wordvecs3 = self.lookup_table3:forward(words)
    
    
    local rnnout1 = self.rnn1:forward(wordvecs)
    local rnnout2 = self.rnn2:forward(wordvecs2)
    local rnnout3 = self.rnn3:forward(wordvecs3)

    ---concat/attention
    local scores_tmp = self.out:forward(  torch.cat( {rnnout1, rnnout2,rnnout3} ,3)    )
    --sum
    --local scores_tmp = self.out:forward(  rnnout1 + rnnout2 + rnnout3     )
    
    local scores = scores_tmp[1]:view(N, -1)

    local idx = nil
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):view(-1):long()
    end
    seq[{{}, t}]:copy(idx)
  end

  -- After sampling stop remembering states - LSTM1
  for i = 1, #self.rnn1 do
    local layer = self.rnn1:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end
  -- After sampling stop remembering states - LSTM2
  for i = 1, #self.rnn2 do
    local layer = self.rnn2:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end
  -- After sampling stop remembering states - LSTM3
  for i = 1, #self.rnn3 do
    local layer = self.rnn3:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end

  self.output = seq
  return self.output
end

function LM:parameters()
  return self.net:parameters()
end

function LM:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end

function LM:clearState()
  self.net:clearState()
end
