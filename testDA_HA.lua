require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
-- require 'gnuplot'
require 'lfs'
require 'image'
-- require 'util.misc'   -- miscellaneous
require 'auxDA'       -- Bayesian Filtering specific auxiliary methods

matio = require 'matio'

nngraph.setDebug(true)  -- uncomment for debug mode
torch.setdefaulttensortype('torch.FloatTensor')

local RNN = require 'model.LSTMDA' -- RNN model for BF
local model_utils = require 'util.model'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a simple trajectory model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-DA_model','HA-11-1_mt2_r128_l1_d2_m5','model name')
cmd:option('-test_file','','test file name')
cmd:option('-M',5,'number of measurements')
cmd:option('-Cur_Frame',2,'')

-- data process
cmd:option('-norm_type',1,'1=max min, 2=zero mean')
cmd:text()

-- parse input params
DAopt = cmd:parse(arg) --把cmd中的参数传入opt

ggtime = torch.Timer() -- start global timer

-- create auxiliary directories (or make sure they exist)附加目录
-- createAuxDirs()
-- checkCuda()       -- check for cuda availability
-- torch.manualSeed(DAopt.seed)  -- manual seed for deterministic runs确定性运行的手动种子

---------------------------------------------------------
------------------Loading Model--------------------------
---------------------------------------------------------
DAopt.DA_model = lfs.currentdir() .. '/../'..'bin/'..DAopt.DA_model..'.t7'

if not lfs.attributes(DAopt.DA_model, 'mode') then
  print('Error: File ' .. DAopt.DA_model.. ' does not exist.?')
end
checkpoint_da = torch.load(DAopt.DA_model)

protos = checkpoint_da.protos
protos.rnn:evaluate()

opt = checkpoint_da.opt
-- for k,v in pairs(checkpoint_da['opt']) do
--   print(k,v)
-- end
-- abort()

miniBatchSize = 1
opt.mini_batch_size = 1
opt.data_valid = 1

stateDim = opt.state_dim

opt.gpuid = -1
opt.opencl = 0
TRAINING = false
TESTING = true

DAinit_state = getDAInitState(opt,1)
  -- 1 : FloatTensor - size: 1x128
  -- 2 : FloatTensor - size: 1x128
-- print(DAinit_state) abort()
---------------------------------------------------------
------------------Getting DATA  -------------------------
---------------------------------------------------------
dataFn = lfs.currentdir() .. '/../testLSTM_DATA/DAs_'..DAopt.Cur_Frame..'.mat'
-- if not lfs.attributes(dataFn, 'mode') then
--   print('Error: File ' .. dataFn.. ' does not exist.?')
-- end
-- print(dataFn)
local loaded = matio.load(dataFn)
-- DAs 是归一化后的数据
local DAs = loaded.DAs:float()  -- FloatTensor - size: 5x10
-- print(DAs) abort()

---------------------------------------------------------
------------------Main Prediction------------------------
---------------------------------------------------------
function getInput(t, lstm_state)
  local input = {}

  for i = 1,#lstm_state[t-1] do table.insert(input,lstm_state[t-1][i]) end
  
  local aDas = DAs[{{t},{}}]:clone()
  table.insert(input,aDas)
  return input
end

local lstm_state = {[0] = DAinit_state}
T = DAs:size(1)
local predDA = torch.Tensor(T,T)
local starttime = os.clock(); 
for t = 1, T do
  -- print('lstm_state 1') print(lstm_state)
  local inp = getInput(t, lstm_state)
  
  local lst = protos.rnn:forward(inp)  -- do one forward tick
  predDA[{{t},{}}] = lst[opt.sIndex]:clone()
  -- print(lst) 
  -- print(predDA)

  lstm_state[t] = {}
  for i=1,#DAinit_state do table.insert(lstm_state[t], lst[i]) end
  -- print('lstm_state 2') print(lstm_state)
  -- abort()
end
local endtime = os.clock();                           --> os.clock()用法


fileName = lfs.currentdir() .. '/../out/PredDA_f'..DAopt.Cur_Frame..'.mat'
local runtime = endtime - starttime
matio.save(fileName,{predDA = predDA,runtime = runtime})


-- print(predDA)
-- abort()

---------------------------------------------------------
------------------Building Model-------------------------
---------------------------------------------------------

