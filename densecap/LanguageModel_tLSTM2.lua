require 'nn'
require 'torch-rnn'
require 'nngraph'

local utils = require 'densecap.utils'

local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')

function LM:__init(opt)
  parent.__init(self)

  opt = opt or {}
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.image_vector_dim = utils.getopt(opt, 'image_vector_dim')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.seq_length = utils.getopt(opt, 'seq_length')/2 
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.idx_to_token = utils.getopt(opt, 'idx_to_token')
  self.dropout = utils.getopt(opt, 'dropout', 0)
  
  local W, D = self.input_encoding_size, self.image_vector_dim
  local V, H = self.vocab_size, self.rnn_size
  local S = 6 --- size of spatial feature
  
  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()  
  
  self.image_encoder:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  self.image_encoder:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  --self.image_encoder:add(nn.View(-1):setNumInputDims(3))-- (B, 512,7,7)->(B,25088)
  --self.image_encoder:add(nn.Linear(25088,512))
  --self.image_encoder:add(nn.ReLU(true))
  --self.image_encoder:add(nn.Dropout(0.5))
  --self.image_encoder:add(nn.Linear(512,512))
  --self.image_encoder:add(nn.ReLU(true))
  --self.image_encoder:add(nn.Dropout(0.5))

  
  self.image_encoder_real = nn.Sequential()
  
  self.spatial_encoder = nn.Sequential()
  self.spatial_encoder:add(nn.Linear(6,64))
  self.spatial_encoder:add(nn.ReLU(true))
  
  local image_encoder12 = nn.ParallelTable()---merge imvec and mask activation
  image_encoder12:add(self.image_encoder)
  image_encoder12:add(self.spatial_encoder)
  
  
  self.image_encoder_real:add(image_encoder12)
  self.image_encoder_real:add(nn.JoinTable(2,2))
  
  self.image_encoder_real:add(nn.Linear(512+64,W))
  self.image_encoder_real:add(nn.ReLU(true))
  self.image_encoder_real:add(nn.View(1, -1):setNumInputDims(1)) 

  
  -----------------------2-----------------
  self.image_encoder2 = nn.Sequential()  
  
  --self.image_encoder2:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  --self.image_encoder2:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  --self.image_encoder2:add(nn.View(-1):setNumInputDims(3))-- (B, 512,7,7)->(B,25088)
  --self.image_encoder2:add(nn.Linear(25088,512))
  --self.image_encoder2:add(nn.ReLU(true))
  
  --self.image_encoder2:add(nn.SpatialConvolution(512, 16, 1, 1))
  --self.image_encoder2:add(nn.ReLU(true))
  --self.image_encoder2:add(nn.View(-1):setNumInputDims(3))-- (B, 16,7,7)->(B,784)
  
  
  self.image_encoder2:add(nn.Linear(4096, W))----!!!222 (B,4096)-> (B,512)
  self.image_encoder2:add(nn.ReLU(true))
  self.image_encoder2:add(nn.View(1, -1):setNumInputDims(1))
  -----------------------------------------
  self.image_encoder3 = nn.Sequential()  
  
  --self.image_encoder3:add(nn.SpatialAveragePooling(7,7))------!!!!2222(B, 512,7,7)->(B, 512,1,1)
  --self.image_encoder3:add(nn.View(-1):setNumInputDims(3))--!!!222 (B, 512,1,1) -> (B,512)
  
  --self.image_encoder3:add(nn.View(-1):setNumInputDims(3))-- (B, 512,7,7)->(B,25088)
  --self.image_encoder3:add(nn.Linear(25088,512))
  --self.image_encoder3:add(nn.ReLU(true))
  
  --self.image_encoder3:add(nn.SpatialConvolution(512, 16, 1, 1))
  --self.image_encoder3:add(nn.ReLU(true))
  --self.image_encoder3:add(nn.View(-1):setNumInputDims(3))-- (B, 16,7,7)->(B,784)
  
  self.image_encoder3:add(nn.Linear(4096, W))----!!!222 (B,4096)-> (B,512)
  self.image_encoder3:add(nn.ReLU(true))
  self.image_encoder3:add(nn.View(1, -1):setNumInputDims(1))

  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  local V, W = self.vocab_size, self.input_encoding_size
  self.lookup_table = nn.LookupTable(V + 2, W)
  self.lookup_table2 = nn.LookupTable(V + 2, W) 
  self.lookup_table3 = nn.LookupTable(V + 2, W)
  
  -- Change this to sample from the distribution instead
  self.sample_argmax = true

  
  ------------------------------------------------------------------------------
  -- self.rnn maps wordvecs of shape N x T x W to word probabilities
  -- of shape N x T x (V + 1)
  self.rnn1 = nn.Sequential()---------LSTM1
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn1:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn1:add(nn.Dropout(self.dropout))
    end
  end
  local parallel1 = nn.ParallelTable()
  if S==0 then
    parallel1:add(self.image_encoder)
  elseif S==6 then
    parallel1:add(self.image_encoder_real)
  else
    dbg()
  end
  parallel1:add(self.start_token_generator)
  parallel1:add(self.lookup_table)
  self.input1 = nn.Sequential()
  self.input1:add(parallel1)
  self.input1:add(nn.JoinTable(1, 2))
  self.input1:add(self.rnn1)
  ------------------------------------------------------------------------------
  self.rnn2 = nn.Sequential()---------LSTM2
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn2:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn2:add(nn.Dropout(self.dropout))
    end
  end
  local parallel2 = nn.ParallelTable()
  parallel2:add(self.image_encoder2)
  parallel2:add(self.start_token_generator)
  parallel2:add(self.lookup_table2)
  self.input2 = nn.Sequential()
  self.input2:add(parallel2)
  self.input2:add(nn.JoinTable(1, 2))
  self.input2:add(self.rnn2)
  ------------------------------------------------------------------------------
  self.rnn3 = nn.Sequential()---------LSTM3
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn3:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn3:add(nn.Dropout(self.dropout))
    end
  end
  local parallel3 = nn.ParallelTable()
  parallel3:add(self.image_encoder3)
  parallel3:add(self.start_token_generator)
  parallel3:add(self.lookup_table3)
  self.input3 = nn.Sequential()
  self.input3:add(parallel3)
  self.input3:add(nn.JoinTable(1, 2))
  self.input3:add(self.rnn3)
  ------------------------------------------------------------------------------
  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)

  
  local combine = nn.ParallelTable()---merge imvec and mask activation
  combine:add(self.input1)
  combine:add(self.input2)
  combine:add(self.input3)
  
  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  self.net:add(combine)
  
  --self.net:add(nn.JoinTable(2, 2))----(concat)
  self.net:add(nn.CAddTable())---(sum),
  --self.net:add(nn.CMulTable())---(mul)


  self.out=nn.Sequential()

  self.out:add(self.view_in)-- (B,1,3*H)->(B,3*H)
  self.out:add(nn.Linear(H, H))
  self.out:add(nn.ReLU(true))
  self.out:add(nn.Dropout(0.5))

  
   ----------------------------------------------------------
  self.view_out2 = nn.View(1, -1):setNumInputDims(2)
  local cap_branch = nn.Sequential()--branch for word output
  cap_branch:add(nn.Linear(H, V + 1))
  cap_branch:add(self.view_out)
  
  local cls_branch = nn.Sequential()
  cls_branch:add(nn.Linear(H, 3))-- parts class
  cls_branch:add(self.view_out2)
  
  local branch = nn.ConcatTable()
  branch:add(cap_branch)
  branch:add(cls_branch)
  
  self.out:add(branch)
  --------------- -----------------------------------------

  self.net:add(self.out)
  
  self:training()
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

  if gt_sequence:nElement() > 0 then  
    self.gt_parts = gt_sequence[{{},{16,30}}]
    gt_sequence = gt_sequence[{{},{1,15}}]
    
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local N, T = gt_sequence:size(1), gt_sequence:size(2)
    self._gt_with_start = gt_sequence.new(N, T + 1)
    self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)
    self._gt_with_start[{{}, {2, T + 1}}]:copy(gt_sequence)
    local mask = torch.eq(self._gt_with_start, 0)
    self._gt_with_start[mask] = self.NULL_TOKEN
    
    -- Reset the views around the nn.Linear
    self.view_in:resetSize(N * (T + 2), -1)
    self.view_out:resetSize(N, T + 2, -1)
    self.view_out2:resetSize(N, T+2 , -1)
    
    self.output = self.net:updateOutput{{{union_vectors,spatial}, self._gt_with_start},{subj_vectors, self._gt_with_start},{obj_vectors, self._gt_with_start}}
    self._forward_sampled = false
    return self.output
  else
    self._forward_sampled = true
    return self:sample(union_vectors, subj_vectors, obj_vectors, spatial)
  end
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

function LM:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput)
  end
  return self.gradInput
end


function LM:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

function LM:backward(input, gradOutput, scale)
  assert(self._forward_sampled == false, 'cannot backprop through sampling')
  assert(scale == nil or scale == 1.0)
  self.recompute_backward = false

  local net_input
  if gradOutput == 0 then---if there is no box- pair detected!
      self.gradInput = {input[1]:new():zero(),nil,input[3]:new():zero(), input[4]:new():zero()}
  else

    net_input = { {{input[1]:cuda(),input[3]}, self._gt_with_start},{input[4][1]:cuda(), self._gt_with_start},{input[4][2]:cuda(), self._gt_with_start} }
     local tmp = self.net:backward(net_input, gradOutput, scale)
     self.gradInput = {tmp[1][1][1],nil , tmp[1][1][2],{tmp[2][1],tmp[3][1]}}
  end

  self.gradInput[2] = input[2]:new():zero()
  return self.gradInput
end

function LM:parameters()
  return self.net:parameters()
end

function LM:training()
  parent.training(self)
  self.net:training()
end

function LM:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end

function LM:clearState()
  self.net:clearState()
end
