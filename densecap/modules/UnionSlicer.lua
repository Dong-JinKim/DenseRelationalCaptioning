local layer, parent = torch.class('nn.UnionSlicer', 'nn.Module')

function layer:__init()
  parent.__init(self)
  self.grad_features = torch.Tensor()
  self.grad_gt_features = torch.Tensor()
end

function layer:updateOutput(input)
  local features = input[1]
  local label_idx = input[2]
  
  if label_idx:nElement() == 0 then
    self.output = features--.new(1):zero()
  else
    self.output = { features:index(1,label_idx[{1,{}}]) , features:index(1,label_idx[{2,{}}])  }
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
    self.grad_features:resizeAs(features):zero()
    self.gradInput = {self.grad_features, self.grad_gt_features}
  end

  return self.gradInput
end
