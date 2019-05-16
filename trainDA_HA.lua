require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
require 'lfs'
require 'image'
require 'util.misc'   -- miscellaneous
require 'auxDA'       -- Bayesian Filtering specific auxiliary methods

matio = require 'matio'  --导入模块 

nngraph.setDebug(true)  -- uncomment for debug mode
torch.setdefaulttensortype('torch.FloatTensor')

local RNN = require 'model.LSTMDA' -- RNN model for BF local为局部变量
local model_utils = require 'util.model'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a simple trajectory model')
cmd:text()
cmd:text('Options')
-- model params cmd.option(name,default,help) 将default参数存储在name中 
cmd:option('-modelName','HA-1-2','model name')
cmd:option('-model_index',2,'1=RNN, 2=LSTM ')
cmd:option('-rnn_size', 128, 'size of pre LSTM internal state')
cmd:option('-num_layers',1,'number of layers in the RNN / LSTM')--RNN/LSTM中的隐含层数
cmd:option('-max_n',1,'state dimension (1-4)')--
cmd:option('-max_m',5,'一个时刻的测量数')-- 可扩展
cmd:option('-state_dim',2,'state dimension (1-4)')--状态维数
cmd:option('-dropout',0.02,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-lambda',1,'pred loss weighting') --损失权重：预测
-- cmd:option('-norm_type',2,'1= max-min, 2=std-dev')

-- data process
cmd:option('-data_train',2000,'number of process scenes to augment training data with') --用于增加训练数据事件的数量
cmd:option('-data_valid',2,'number of synthetic scenes to augment validation data with')
cmd:option('-norm_type',1,'1=max min, 2=zero mean')

-- optimation params
cmd:option('-seed',10,'torch manual random number generator seed')
cmd:option('-max_epochs',35000,'number of full passes through the training data')
-- cmd:option('-temp_win',3, 'temporal window history')--时间窗口历史
cmd:option('-mini_batch_size',10,'mini-batch size')
cmd:option('-lrng_rate',1e-03,'learning rate')
cmd:option('-lrng_rate_decay',0.99,'learning rate decay')
cmd:option('-lrng_rate_decay_every',2000,'learn rate decay')
cmd:option('-decay_rate',0.97,'decay rate for rmsprop')
cmd:option('-grad_clip',.1,'clip gradients at this value')
-- cmd:option('-lrng_rate_decay_after',-1,'in number of epochs, when to start decaying the learning rate')

cmd:option('-eval_val_every',3000,'every how many iterations should we evaluate on validation data?')
cmd:option('-print_every',500,'how many steps/mini-batches between printing out the loss')
cmd:option('-plot_progress_every',999,'how many steps/mini-batches between printing the pred process')

-- GPU/CPU
cmd:option('-gpuid',1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params  分析输入参数
opt = cmd:parse(arg) --把cmd中的参数传入opt  在后面使用opt调用前面设置的参数

ggtime = torch.Timer() -- start global timer

-- create auxiliary directories (or make sure they exist)附加目录
createAuxDirs()
checkCuda()       -- check for cuda availability
torch.manualSeed(opt.seed)  -- manual seed for deterministic runs确定性运行的手动种子

-- global vars
TRAINING = true
TRAINING_DA = true
-- TRAINING_DA100 = true
-- TRAINING_DA300 = false
-- TRAINING_DA450 = false
stateDim = opt.state_dim
miniBatchSize = opt.mini_batch_size


---------------------------------------------------------
------------------Building Model-------------------------
---------------------------------------------------------
-- opt.nClasses = opt.max_m + 1
if opt.model_index ==1 then opt.model = 'rnn'     --确定模型为rnn或者lstm
else opt.model = 'lstm' end

-- input's index
opt.pcIndex = 1
opt.phIndex = opt.pcIndex + 1
opt.sIndex = opt.phIndex + 1
opt.in_size = opt.state_dim * opt.max_m      -- 2×5

-- output's index
opt.nHiddenInputs = 1
if opt.model=='lstm' then opt.nHiddenInputs = 2 end
opt.predDAIndex = opt.num_layers*opt.nHiddenInputs+1

modelName = opt.modelName

opt.modelParams = {'model_index', 'rnn_size', 'num_layers','state_dim', 'max_m'}

-- TODO remove these globals
modelParams = opt.modelParams

print('1.Creating a LSTM model.')
protos = {}                    
protos.rnn = RNN.rnn(opt)           --rnn.rnn()返回nn.gModule(inputs, outputs) outputs中有(c,h)和关联概率
local kl = nn.DistKLDivCriterion() --KL散度 度量两个函数的相似程度
local mse = nn.MSECriterion()      --均方差
protos.criterion = nn.ParallelCriterion()--返回一个是其他损失的和的损失
protos.criterion:add(mse, opt.lambda)   --对mse加权求和 lua中:意思是省略函数的第一个self参数

-- ship the model to the GPU if desired
for k,v in pairs(protos) do v = dataToGPU(v) end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
print('2.Number of parameters in the model: ' .. params:nElement())

print('3.Cloning '..opt.max_m..' times...')      --与一个时刻的测量数有关 
clones = {}                                      --每个目标都创建一个模型
for name,proto in pairs(protos) do
  print('\t   cloning ' .. name)
  clones[name] = model_utils.clone_many_times(proto, opt.max_m, not proto.parameters)
end
print('   ...done')
-- print(clones)
-- abort()
init_state = getDAInitState(opt, miniBatchSize)   --在auxDA.lua里 初始状态全置0 
val_init_state = getDAInitState(opt, 1)
-- print(init_state) 
-- print(val_init_state)
-- abort()

---------------------------------------------------------
---------------Data Preprocessing------------------------
---------------------------------------------------------
-- local trSeqTable = {'v2_z4_z1','a1_a5_a8','a2_v5_z9_z6','a8_z3_z1_v7_z5','v2_z4','v6_z8_z9_v7','v6_z8_z9_z6','v8_z6',
--                     'z1_z3_a2','z1_z3_z5','z1_z3_z5_z6','z1_z8_z4','z4_v2_z5_v4','z6_z8','z7_z6'}

-- local valSeqTable={'v2_z4_z1','v8_z6_z1','z4_a6_z5_v4','z4_z7_z5_a8','a8_z3_z1_v7_z5'}

local trSeqTable = {'MT_N5_M5_1','MT_N5_M5_2','MT_N5_M5_3','MT_N5_M5_4','MT_N5_M5_5','MT_N5_M5_6','MT_N5_M5_7','MT_N5_M5_8','MT_N5_M5_9','MT_N5_M5_10',
                    'MT_N5_M5_11','MT_N5_M5_12','MT_N5_M5_13','MT_N5_M5_14','MT_N5_M5_15','MT_N5_M5_16','MT_N5_M5_17','MT_N5_M5_18','MT_N5_M5_19','MT_N5_M5_20',
                    'MT_N5_M5_21','MT_N5_M5_22','MT_N5_M5_23','MT_N5_M5_24','MT_N5_M5_25','MT_N5_M5_26','MT_N5_M5_27','MT_N5_M5_28','MT_N5_M5_29','MT_N5_M5_30',
                    'MT_N5_M5_31','MT_N5_M5_32','MT_N5_M5_33','MT_N5_M5_34','MT_N5_M5_35','MT_N5_M5_36','MT_N5_M5_37','MT_N5_M5_38','MT_N5_M5_39','MT_N5_M5_40',
                    'MT_N5_M5_41','MT_N5_M5_42','MT_N5_M5_43','MT_N5_M5_44','MT_N5_M5_45','MT_N5_M5_46','MT_N5_M5_47','MT_N5_M5_48','MT_N5_M5_49','MT_N5_M5_50',}
-- trSeqTable={'curve1','line1'}
local trSTab, trDAgtTab = preDAData('train', trSeqTable)                 --在zhdata.lua里面 gt = target 

-- print('trSTab') print(trSTab)        -- data_train: mbSize * (M * M * 3)
-- print('trDAgtTab') print(trDAgtTab)  -- data_train: mbSize * (M * (M+1))
-- print(#trSTab)
-- abort()

--------------------------------------------------------------------------
--- To compute loss and gradients, we construct tables with  找预测值和真实值
-- predicted values (input) and ground truth values (target)
function getPredAndGTTables( predDA, t)
  local input, target = {}, {}

  -- DAgts = trDAgtTab[seqCnt]:clone()      -- 2维：mbSize * (M * (M+1)) = mbSize * (opt.max_m * opt.nClasses)
  local oneDagt = nil
  local begin = opt.max_m * (t-1) + 1
  oneDagt = DAgts:narrow(2,begin,opt.max_m):clone()  -- 2维：mbSize * (1 * (M+1) = mbSize * (1 * opt.nClasses) = mbSize * opt.nClasses
  -- print('oneDagt') print(oneDagt)
  
  table.insert(input,predDA)
  table.insert(target, oneDagt)
  for k,v in pairs(input) do input[k] = dataToGPU(v) end
  for k,v in pairs(target) do target[k] = dataToGPU(v) end
  -- print('target') print(target)
  return input, target
end


---------------------------------------------------------
---------LOSS AND GRADIENT COMPUTATION-------------------
---------------------------------------------------------

-- local permPlot = math.random(tabLen(trTracksTab)) -- always plot this training seq

local trTabL = opt.data_train      --2000
local seqCnt=0

function feval()  --feval在matlab中用于计算某点的函数值 返回loss, grad_params
  grad_params:zero()

  -- 选择训练的数据序列
  seqCnt=seqCnt+1     --1
  if seqCnt > trTabL then seqCnt = 1 end --seqCnt>2000时，重置为1

  -- These are global data for this one iteration
  Ss = trSTab[seqCnt]:clone()        -- 2维：mbSize * (M * M * 3)  clone之后修改副本，原来的不会变
  DAgts = trDAgtTab[seqCnt]:clone()  -- 2维：mbSize * (M * (M+1))
  -- print('Ss') print(Ss)
  -- print('DAgts') print(DAgts)
  -- abort()

  ----- FORWARD ----
  local initStateGlobal = clone_list(init_state) --init_state = getDAInitState(opt, miniBatchSize)
  local rnn_state = {[0] = initStateGlobal}
  local predictions = {}

  local loss = 0
  local predDA= {}

  local statePred = {}
  local T = opt.max_m      --5

  TRAINING = true -- flag to be used for input
  -- print('forwarding..........')
  for t=1,T do
    -- print('t:'..t)
    --由于torch每个模块均有train参数，当其为true时进行训练，当期为false时进行测试。
    clones.rnn[t]:training()-- set flag for dropout

    --获取一个时间步骤中所有的输入
    --若为第一帧，take detections as states
    local rnninp, rnn_state = getDAInput(t, rnn_state, predictions)    -- get combined RNN input table  链接检测和pre
    -- rnninp:
    --   1 : FloatTensor - size: 3x128
    --   2 : FloatTensor - size: 3x128
    --   3 : FloatTensor - size: 3x15
    -- print('rnninp:')print(rnninp)
    -- print('rnninp[1]') print(rnninp[1]) print('rnninp[2]') print(rnninp[2])
    -- print('rnninp[3]') print(rnninp[3]) 

    local lst = clones.rnn[t]:forward(rnninp) -- do one forward tick
    -- lst:
    --   1 : FloatTensor - size: 3x128
    --   2 : FloatTensor - size: 3x128
    --   3 : FloatTensor - size: 3x6
    -- print('lst:') print(lst)
    -- print('lst[1]') print(lst[1]) print('lst[2]') print(lst[2])
    -- print('lst[3]') print(lst[3])  abort()

    predictions[t] = lst 

    -- update hidden state   
    rnn_state[t] = {}
    for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output 提取状态

    -- prediction at time t (for t+1)   t时刻的预测关联
    predDA[t] = decodeDA(predictions, t)
    -- print('predDA[t]') print(predDA)
    -- abort()
    local input, target = getPredAndGTTables(predDA[t], t) 
    -- print('input') print(input[1])
    -- print('target') print(target[1])
    -- abort()

    local tloss = clones.criterion[t]:forward(input, target) -- compute loss for one frame
    loss = loss + tloss
  end
  loss = loss / T -- make average over all frames
  -- print('loss') print(loss) 
  -- abort()

  ------ PLOT PROGRESS-----------
  if globiter % opt.plot_progress_every == 0 then
    plotTraingDAProcess(predDA, DAgts, T)
  end

  ------ BACKWARD ------
  local rnn_backw = {}
  -- gradient at final frame is zero
  local drnn_state = {[T] = clone_list(init_state, true)} -- true = zero the clones
  -- print('drnn_state1') print(drnn_state)
  -- print('backwarding..........')
  for t=T,1,-1 do --从T变化到1以-1为步长 T=5
    -- print('t..'..t)
    local input, target = getPredAndGTTables(predDA[t], t)  --找到预测值和真实值
    -- print('drnn_state1') print(drnn_state)  
    local dl = clones.criterion[t]:backward(input,target) 
    -- print('dl') print(dl)
    for dd=1,#dl do
      table.insert(drnn_state[t], dl[dd]) -- gradient of loss at time t
    end
    -- print('drnn_state2') print(drnn_state)
    local rnninp, rnn_state = getDAInput(t, rnn_state, predictions)    -- get combined RNN input table
    -- print('rnninp:') print(rnninp)
    -- print('rnninp[1]') print(rnninp[1])
    -- print('rnninp[2]') print(rnninp[2])
    -- print('rnninp[3]') print(rnninp[3])
    -- abort()
    local dlst = clones.rnn[t]:backward(rnninp, drnn_state[t])  
    -- print('dlst')print(dlst)
    drnn_state[t-1] = {}
    
    drnn_state[t-1][1]=dlst[1]
    drnn_state[t-1][2]=dlst[2]
    -- print('drnn_state3') print(drnn_state)
    -- abort()

    grad_params:clamp(-opt.grad_clip, opt.grad_clip) --把梯度参数限制在一个范围以内

  end
  -- abort()
  return loss, grad_params

end
---------------------------------------------------------
-------------------Optimation----------------------------
---------------------------------------------------------
train_losses = {}
val_losses ={}
min_cal_loss = 0
printTrainingHeadline()
local optim_state = {learningRate = opt.lrng_rate, alpha = opt.decay_rate} --学习率和衰减率
local glTimer = torch.Timer()
for i = 1, opt.max_epochs do   --训练35000次
  local epoch = i
  globiter = i
  --opt.random_epoch=0
  -- if i>1 and (i-1)%opt.data_train==0 and opt.random_epoch~=0 then
  --   trTracksTab, trDetsTab = preData('train', trSeqTable)
  -- end

  local timer = torch.Timer() 
  local _, loss = optim.rmsprop(feval, params, optim_state)   --require optim 参数为(feval,params，学习率和衰减率)
  -- print('loss..'..loss[1])
  -- abort()
  local time = timer:time().real

  local train_loss = loss[1] -- the loss is inside a list, pop it
  train_losses[i] = train_loss

  -- exponential learning rate decay
  if i % opt.lrng_rate_decay_every == 0 and opt.lrng_rate_decay < 1 then
    -- print('decreasing learning rate')
    local decay_factor = opt.lrng_rate_decay
    optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
    --         end
  end

  if i % opt.print_every == 0 then  --每隔500输出一次 
    printTrainingStats(i, opt.max_epochs, train_loss, grad_params:norm() / params:norm(),
      time, optim_state.learningRate, glTimer:time().real)
  end

  -- if i % opt.eval_val_every == 0 then
  --   val_loss = eval_val()
  --   val_losses[i] = math.max(val_loss, 1e-5)
  --   if min_cal_loss == 0 then
  --     min_cal_loss = val_loss
  --   end
  --   if val_loss < min_cal_loss then
  --     min_cal_loss = val_loss
  --     local savefile = getCheckptFilename(opt.modelName, opt, opt.modelParams, 1, i, val_loss)
  --     saveCheckpoint(savefile, protos, opt, train_losses, glTimer:time().real, opt.max_epochs)
  --   end
  -- end

end

-- abort()

-- plotLoss(opt.max_epochs, opt.eval_val_every)
-- save global checkpt
local savefile = getCheckptFilename(opt.modelName, opt, opt.modelParams, 0)
saveCheckpoint(savefile, protos, opt, train_losses, glTimer:time().real, opt.max_epochs)

print(string.format('%20s%10.2f%7s','total time',ggtime:time().real,''))
