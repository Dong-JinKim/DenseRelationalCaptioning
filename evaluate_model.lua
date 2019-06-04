require 'torch'
require 'nn'
require 'cudnn'

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'

local utils = require 'densecap.utils'
local eval_utils = require 'eval.eval_utils_imRecall'

local cmd = torch.CmdLine()
cmd:option('-checkpoint',
  'checkpoint_VGlongv3_tLSTM_MTL2_1e6.t7',
  'The checkpoint to evaluate')
cmd:option('-data_h5', '', 'The HDF5 file to load data from; optional.')
cmd:option('-data_json', '', 'The JSON file to load data from; optional.')
cmd:option('-gpu', 0, 'The GPU to use; set to -1 for CPU')
cmd:option('-use_cudnn', 1, 'Whether to use cuDNN backend in GPU mode.')
cmd:option('-split', 'val', 'Which split to evaluate; either val or test.')
cmd:option('-max_images', -1, 'How many images to evaluate; -1 for whole split')
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-num_proposals', 50)
local opt = cmd:parse(arg)


-- First load the model
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
print 'Loaded model'

local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
print(string.format('Using dtype "%s"', dtype))

model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh=opt.rpn_nms_thresh,
  final_nms_thresh=opt.final_nms_thresh,
  max_proposals=opt.num_proposals,
}

if opt.data_h5 == '' then
  opt.data_h5 = checkpoint.opt.data_h5
end
if opt.data_json == '' then
  opt.data_json = checkpoint.opt.data_json
end
local loader = DataLoader(opt)

local eval_kwargs = {
  model=model,
  loader=loader,
  split=opt.split,
  max_images=opt.max_images,
  dtype=dtype,
}
local eval_results = eval_utils.eval_split(eval_kwargs)
