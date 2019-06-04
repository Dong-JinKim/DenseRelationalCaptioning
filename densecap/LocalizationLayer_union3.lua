require 'torch'
require 'nn'

require 'densecap.modules.OurCrossEntropyCriterion'
require 'densecap.modules.BilinearRoiPooling'
require 'densecap.modules.ReshapeBoxFeatures'
require 'densecap.modules.ApplyBoxTransform'
require 'densecap.modules.InvertBoxTransform'
require 'densecap.modules.BoxSamplerHelper'
require 'densecap.modules.RegularizeLayer'
require 'densecap.modules.MakeAnchors'
require 'densecap.modules.Pairs_RPN'

local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'

local layer, parent = torch.class('nn.LocalizationLayer', 'nn.Module')

function layer:__init(opt)
  parent.__init(self)
end

function layer:parameters()
  return self.nets.rpn:parameters()
end

function layer:setImageSize(image_height, image_width)
  self.image_height = image_height
  self.image_width = image_width
  self._called_forward_size = false
  self._called_backward_size = false
end


function layer:setGroundTruth(gt_boxes, gt_labels)
  self.gt_boxes = gt_boxes
  self.gt_labels = gt_labels
  self._called_forward_gt = false
  self._called_backward_gt = false
end


function layer:reset_stats()
  self.stats = {}
  self.stats.losses = {}
  self.stats.times = {}
  self.stats.vars = {}
end


function layer:clearState()
  self.timer = nil
  self.rpn_out = nil
  self.rpn_boxes = nil
  self.rpn_anchors = nil
  self.rpn_trans = nil
  self.rpn_scores = nil
  self.pos_data = nil
  self.pos_boxes = nil
  self.pos_anchors = nil
  self.pos_trans = nil
  self.pos_target_data = nil
  self.pos_target_boxes = nil
  self.pos_target_labels = nil
  self.neg_data = nil
  self.neg_scores = nil
  self.roi_boxes:set()
  self.nets.rpn:clearState()
  self.nets.roi_pooling:clearState()
end


function layer:timeit(name, f)
  self.timer = self.timer or torch.Timer()
  if self.timing then
    cutorch.synchronize()
    self.timer:reset()
    f()
    cutorch.synchronize()
    self.stats.times[name] = self.timer:time().real
  else
    f()
  end
end


function layer:setTestArgs(args)
  args = args or {}
  self.test_clip_boxes = utils.getopt(args, 'clip_boxes', true)
  self.test_nms_thresh = utils.getopt(args, 'nms_thresh', 0.7)
  self.test_max_proposals = utils.getopt(args, 'max_proposals', 300)
end


function layer:updateOutput(input)
  return self:_forward_test(input)
end


function layer:_forward_test(input)
  local cnn_features = input
  local arg = {
    clip_boxes = self.test_clip_boxes,
    nms_thresh = self.test_nms_thresh,
    max_proposals = self.test_max_proposals
  }

  assert(self.image_height and self.image_width and not self._called_forward_size,
         'Must call setImageSize before each forward pass')
  self._called_forward_size = true

  local rpn_out
  self:timeit('rpn:forward_test', function()
    rpn_out = self.nets.rpn:forward(cnn_features)
  end)
  local rpn_boxes, rpn_anchors = rpn_out[1], rpn_out[2]
  local rpn_trans, rpn_scores = rpn_out[3], rpn_out[4]
  local num_boxes = rpn_boxes:size(2)
  
  local valid
  if arg.clip_boxes then
    local bounds = {
      x_min=1, y_min=1,
      x_max=self.image_width,
      y_max=self.image_height
    }
    rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'xcycwh')

    local function clamp_data(data)
      assert(data:size(1) == 1, 'must have 1 image per batch')
      assert(data:dim() == 3)
      local mask = valid:view(1, -1, 1):expandAs(data)
      return data[mask]:view(1, -1, data:size(3))
    end
    rpn_boxes = clamp_data(rpn_boxes)
    rpn_anchors = clamp_data(rpn_anchors)
    rpn_trans = clamp_data(rpn_trans)
    rpn_scores = clamp_data(rpn_scores)

    num_boxes = rpn_boxes:size(2)
  end
  
  local rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(rpn_boxes)

  local rpn_scores_exp = torch.exp(rpn_scores)
  local pos_exp = rpn_scores_exp[{1, {}, 1}]
  local neg_exp = rpn_scores_exp[{1, {}, 2}]
  local scores = (pos_exp + neg_exp):pow(-1):cmul(pos_exp)

  local boxes_scores = scores.new(num_boxes, 5)
  boxes_scores[{{}, {1, 4}}] = rpn_boxes_x1y1x2y2
  boxes_scores[{{}, 5}] = scores
  local idx
  self:timeit('nms', function()
    if arg.max_proposals == -1 then
      idx = box_utils.nms(boxes_scores, arg.nms_thresh)
    else
       idx = box_utils.nms(boxes_scores, arg.nms_thresh, arg.max_proposals)
    end
  end)
  
  local rpn_boxes_nms = rpn_boxes:index(2, idx)[1]
  local rpn_anchors_nms = rpn_anchors:index(2, idx)[1]
  local rpn_trans_nms = rpn_trans:index(2, idx)[1]
  local rpn_scores_nms = rpn_scores:index(2, idx)[1]
  local scores_nms = scores:index(1, idx)
  
  local union_boxes, label_idx,Spatial = nn.Pairs_RPN():forward{rpn_boxes_nms, cnn_features.new(0) }
  
  
  local roi_features
  local union_features
  self:timeit('roi_pooling:forward_test', function()
    self.nets.roi_pooling:setImageSize(self.image_height, self.image_width)
    local tmp_feats = self.nets.roi_pooling:forward{cnn_features[1], torch.cat(rpn_boxes_nms,union_boxes,1)}
    roi_features = tmp_feats[{{1,rpn_boxes_nms:size(1)},{}}]
    union_features = tmp_feats[{{rpn_boxes_nms:size(1)+1,-1},{}}]
  end)
  
  if self.dump_vars then
    local vars = self.stats.vars or {}
    vars.test_rpn_boxes_nms = rpn_boxes_nms
    vars.test_rpn_anchors_nms = rpn_anchors_nms
    vars.test_rpn_scores_nms = scores:index(1, idx)
    self.stats.vars = vars
  end
  
  local empty = roi_features.new()
  self.output = {roi_features, rpn_boxes_nms, empty, empty, union_features,Spatial,label_idx}
  return self.output
end
