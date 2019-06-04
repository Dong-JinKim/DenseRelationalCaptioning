local cjson = require 'cjson'

local utils = {}



function utils.setup_gpus(gpu, use_cudnn)
  local dtype = 'torch.FloatTensor'
  local actual_use_cudnn = false
  if gpu >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(gpu + 1)
    dtype = 'torch.CudaTensor'
    if use_cudnn == 1 then
      require 'cudnn'
      actual_use_cudnn = true
    end
  end
  return dtype, actual_use_cudnn
end


function utils.build_loss_string(losses)
  local x = ''
  for k,v in pairs(losses) do
    if k ~= 'total_loss' then
      x = x .. string.format('%s: %.3f, ', k, v)
    end
  end
  x = x .. string.format(' [total: %.3f]', losses.total_loss)
  return x
end

function utils.build_timing_string(timings)
  local x = ''  
  for k,v in pairs(timings) do
    x = x .. string.format('timing %s: %.3fms', k, v*1000) .. '\n'
  end
  return x
end

function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.ensureopt(opt, key)
  if opt == nil or opt[key] == nil then
    error('error: required key ' .. key .. ' was not provided.')
  end
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n 
  end
  return dict
end

function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

utils.__GLOBAL_STATS__ = {}

return utils
