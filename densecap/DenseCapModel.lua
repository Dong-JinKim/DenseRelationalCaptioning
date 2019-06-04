require 'torch'
require 'nn'
require 'nngraph'

require 'densecap.LanguageModel_tLSTM2'
require 'densecap.LocalizationLayer_union3'
require 'densecap.modules.BoxRegressionCriterion'
require 'densecap.modules.BilinearRoiPooling'
require 'densecap.modules.ApplyBoxTransform'
require 'densecap.modules.LogisticCriterion'
require 'densecap.modules.PosSlicer'
require 'densecap.modules.UnionSlicer'

local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'


local DenseCapModel, parent = torch.class('DenseCapModel', 'nn.Module')


function DenseCapModel:__init(opt)
  return 0
end

function DenseCapModel:training()
  parent.training(self)
  self.net:training()
end


function DenseCapModel:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


function DenseCapModel:setTestArgs(kwargs)
  self.nets.localization_layer:setTestArgs{
    nms_thresh = utils.getopt(kwargs, 'rpn_nms_thresh', 0.7),
    max_proposals = utils.getopt(kwargs, 'num_proposals', 50)
  }
  self.opt.final_nms_thresh = utils.getopt(kwargs, 'final_nms_thresh', 0.3)
end


function DenseCapModel:convert(dtype, use_cudnn)
  self:type(dtype)
  if cudnn and use_cudnn ~= nil then
    local backend = nn
    if use_cudnn then
      backend = cudnn
    end
    cudnn.convert(self.net, backend)
    cudnn.convert(self.nets.localization_layer.nets.rpn, backend)
  end
end


function DenseCapModel:updateOutput(input)
  
  assert(input:dim() == 4 and input:size(1) == 1 and input:size(2) == 3)
  local H, W = input:size(3), input:size(4)
  self.nets.localization_layer:setImageSize(H, W)

  if self.train then
    assert(not self._called_forward,
      'Must call setGroundTruth before training-time forward pass')
    self._called_forward = true
  end
  
  self.output = self.net:forward(input)
  
  if not self.train and self.opt.final_nms_thresh > 0 then
    local final_boxes_float = self.output[4]:float()
    local class_scores_float = self.output[1]:float()

    local lm_output_float = self.output[5]:float()
    
    local boxes_scores = torch.FloatTensor(final_boxes_float:size(1), 5)
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
    boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
    boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, 1}])
    
    local idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)
    self.output[4] = final_boxes_float:index(1, idx):typeAs(self.output[4])
    self.output[1] = class_scores_float:index(1, idx):typeAs(self.output[1])
    
    local idx_ = torch.LongTensor(0)
    for i = 1,idx:size(1) do
      for j = 1,idx:size(1) do
        if not(i==j) then
          if i>j then
            idx_ = idx_:cat(torch.LongTensor({ (idx[i]-1)*(boxes_scores:size(1)-1) +  idx[j]   }),1)
          else--i<j
            idx_ = idx_:cat(torch.LongTensor({ (idx[i]-1)*(boxes_scores:size(1)-1) +  idx[j]-1 }),1)
          end
        
        end
      end
    end
      
    if ##idx_ == 0 then
      self.output[5] = idx_:new(idx_:size()):typeAs(self.output[5])
    else
      self.output[5] = lm_output_float:index(1, idx_ ):typeAs(self.output[5])
    end

  end
  return self.output
end


function DenseCapModel:forward_test(input)
  self:evaluate()
  local output = self:forward(input)
  local final_boxes = output[4]
  local objectness_scores = output[1]

  local caption_code = output[5]
  local captions = self.nets.language_model:decodeSequence(caption_code)

  return final_boxes, objectness_scores, captions
end


function DenseCapModel:setGroundTruth(gt_boxes, gt_labels)
  self.gt_boxes = gt_boxes
  self.gt_labels = gt_labels
  self._called_forward = false
  self.nets.localization_layer:setGroundTruth(gt_boxes, gt_labels)
end


function DenseCapModel:getParameters()
  local cnn_params, grad_cnn_params = self.net:get(2):getParameters()
  local fakenet = nn.Sequential()
  fakenet:add(self.net:get(3))
  fakenet:add(self.net:get(4))
  local params, grad_params = fakenet:getParameters()
  return params, grad_params, cnn_params, grad_cnn_params
end


function DenseCapModel:clearState()
  self.net:clearState()
  for k, v in pairs(self.crits) do
    if v.clearState then
      v:clearState()
    end
  end
end

