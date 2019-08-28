local layer, parent = torch.class('nn.UnionSlicer', 'nn.Module')

function layer:__init()
  parent.__init(self)
  self.grad_features = torch.Tensor()
  self.grad_gt_features = torch.Tensor()
  
  self.subj_index = nn.Index(1)
  self.obj_index = nn.Index(1)
end

function layer:updateOutput(input)
  local features = input[1]
  local label_idx = input[2]
  
  if label_idx:nElement() == 0 then
    self.output = features--.new(1):zero()
  else
    local out_subj = self.subj_index:forward{features,label_idx[{1,{}}]}
    local out_obj  = self.obj_index:forward{features,label_idx[{2,{}}]}
    
    self.output = { out_subj , out_obj  }
  end

  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local features = input[1]
  local gt_features = input[2]
  self.grad_gt_features:zero()
  if gt_features:nElement() == 0 then
    self.gradInput = {gradOutput, self.grad_gt_features}
  else
    --self.grad_features:resizeAs(features):zero()
    self.grad_features = self.subj_index:backward({features,gt_features[{1,{}}]},gradOutput[1])[1] + self.obj_index:backward({features,gt_features[{2,{}}]},gradOutput[2])[1]
    self.gradInput = {self.grad_features, self.grad_gt_features}
  end

  return self.gradInput
end
