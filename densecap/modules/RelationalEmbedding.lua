local layer, parent = torch.class('nn.RelationalEmbedding', 'nn.Module')



function layer:__init()
  parent.__init(self)

  self.emb_size = 512

  local emb_subj = nn.Sequential()------------------------------W
  emb_subj:add(nn.Linear(4096,self.emb_size))-----OU(1*B*1024)
  emb_subj:add(nn.ReLU(true))
  emb_subj:add(nn.Unsqueeze(1))-----------------------(B*1024)
  
  local emb_obj = nn.Sequential()-------------------------------U
  emb_obj:add(nn.Linear(4096,self.emb_size))------OW(1*B*1024)
  emb_obj:add(nn.ReLU(true))
  emb_obj:add(nn.Transpose({1,2}))------------------(1*1024*B)
  emb_obj:add(nn.Unsqueeze(1))------------------------(1024*B)
  
  local emb_main = nn.Sequential()------------------------------H
  emb_main:add(nn.Linear(4096,self.emb_size))------OH(1*B*1024)
  emb_main:add(nn.ReLU(true))
  emb_main:add(nn.Unsqueeze(1))------------------------(B*1024)
  
  self.view_out1 = nn.View(1, 1):setNumInputDims(3)
  self.view_out2 = nn.View(1, 1):setNumInputDims(3)
  
  local concat_input = nn.ConcatTable()
  concat_input:add(emb_subj)
  concat_input:add(emb_obj)
  local attNet = nn.Sequential()--------------------computing attention
  attNet:add(concat_input)
  attNet:add(nn.MM())----------------- OW*OU (1*B*B)
  attNet:add(self.view_out1)-------------- (B*B)
  attNet:add(nn.SoftMax())---------- R = softmax(OW*OU)
  attNet:add(nn.Unsqueeze(1))---------(1*B*B)
  
  
  local concat_attending = nn.ConcatTable()
  concat_attending:add(attNet)
  concat_attending:add(emb_main)
  local resNet = nn.Sequential()---------------------computing residual
  resNet:add(concat_attending)
  resNet:add(nn.MM())----------------------(R(OH)) (*B*1024)
  resNet:add(self.view_out2)-------(B*1024)
  resNet:add(nn.Linear(self.emb_size,4096))----------------------fc(R(OH))) (B*4096)
  resNet:add(nn.ReLU(true))
  
  local concat_output = nn.ConcatTable()-----Sum()
  concat_output:add(resNet)
  concat_output:add(nn.Identity())
  self.net = nn.Sequential()
  self.net:add(concat_output)
  self.net:add(nn.CAddTable()) ---- O+fc(R(OH)))

end

function layer:updateOutput(input)
  local features = input
  local B , D = features:size(1), features:size(2)--32, 4096
  
  self.view_out1:resetSize(B,B)
  self.view_out2:resetSize(B,self.emb_size) --- set matrix size

  self.output = self.net:forward(features)----- forward
  
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local features = input
  
  self.grad_features = self.net:backward(features,gradOutput)
  self.gradInput = self.grad_features
  return self.gradInput
end

