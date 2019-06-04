local layer, parent = torch.class('nn.Pairs_RPN', 'nn.Module')
local box_utils = require 'densecap.box_utils'
require 'densecap.modules.BoxIoU'


function layer:__init()
  parent.__init(self)
  self.grad_features = torch.Tensor()
  self.grad_gt_features = torch.Tensor()
end

function layer:updateOutput(input)
  local box_idxs = input[2]
  local boxes = input[1]
  local B = boxes:size(1)
  
  local label_idx = torch.LongTensor(0)
  local union_boxes = boxes.new(0)
  
  local Spatial = torch.CudaTensor(0)
  local spatial = torch.CudaTensor(1,6):zero()
  
  local continue = false
  local IOU = nn.BoxIoU():forward{boxes:view(1,-1,4) ,boxes:view(1,-1,4)}
  for i =1,B do---subj
    for j = 1,B do----obj
      ----conditions to pass------
      if ##box_idxs == 0  then --for test
        if i==j then
          continue = true---- if i==j then pass
        else
          continue = false
        end
      elseif not( (torch.ceil(box_idxs[i]/2)==torch.ceil(box_idxs[j]/2) and box_idxs[i]%2==1 and box_idxs[j]%2==0) ) then--for training
        continue = true
      else
        continue = false
      end   
      
      if not continue then
        local subj_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(boxes[{{i}}])--box coordinates
        local obj_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(boxes[{{j}}])
        
        local x1y1 = torch.cat(subj_x1y1x2y2,obj_x1y1x2y2,1 ):min(1)[{{},{1,2}}]
        local x2y2 = torch.cat(subj_x1y1x2y2,obj_x1y1x2y2,1 ):max(1)[{{},{3,4}}]
        
        local xs = boxes[{i,1}]
        local ys = boxes[{i,2}]
        local ws = boxes[{i,3}]
        local hs = boxes[{i,4}]
        local xo = boxes[{j,1}]
        local yo = boxes[{j,2}]
        local wo = boxes[{j,3}]
        local ho = boxes[{j,4}]
        
        spatial[{1,1}] = (xs-xo)/torch.sqrt(ws*hs)
        spatial[{1,2}] = (ys-yo)/torch.sqrt(ws*hs)
        spatial[{1,3}] = torch.sqrt(wo*ho)/torch.sqrt(ws*hs)
        spatial[{1,4}] = ws/hs
        spatial[{1,5}] = wo/ho
        spatial[{1,6}] = IOU[{1,i,j}]        
        
        local union = box_utils.x1y1x2y2_to_xcycwh( torch.cat(x1y1,x2y2) )--compute union of box
        
        Spatial = Spatial:cat(spatial , 1)
        union_boxes = union_boxes:cat(union,1)
        label_idx = label_idx:cat(torch.LongTensor({ i,j }),2)
      end
    end
  end
  
  return union_boxes,label_idx,Spatial
end

function layer:updateGradInput(input, gradOutput)
  return 0
end


