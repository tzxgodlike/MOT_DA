--- auxKF

--------------------------------------------------------------------------
--- each layer has the same hidden input size
function getDAInitState(opt, miniBatchSize)
  local init_state = {}
  for L=1,opt.num_layers do
    local h_init = torch.zeros(miniBatchSize, opt.rnn_size)
    table.insert(init_state, dataToGPU(h_init:clone()))
    if opt.model == 'lstm' then
      table.insert(init_state, dataToGPU(h_init:clone()))
    end
  end
  return init_state
end

function dataToGPU(data)
--   if true then return data end
  
--   local timerF = torch.Timer()
  if data==nil then return nil end
  data=data:float()
  
  if opt.gpuid >= 0 and opt.opencl == 0 then
    data = data:float():cuda()
  end
  
  if opt.gpuid >= 0 and opt.opencl == 1 then
    data = data:cl()
  end
--   if opt.profiler>0 then profUpdate('dataToGPU', timerF:time().real) end
  return data
end

--------------------------------------------------------------------------
--- get all inputs for one time step
-- @param t		time step
-- @param rnn_state	hidden state of RNN (use t-1)
-- @param predictions	current predictions to use for feed back
function getDAInput(t, rnn_state, predictions )
  local rnninp = {} 

  -- (1) prev_c  and (2) prev_h
  -- print('rnninp,rnn_state[t-1][1]') print(rnninp,rnn_state[t-1][1]) abort()
  for i = 1,#rnn_state[t-1] do table.insert(rnninp,rnn_state[t-1][i]) end
  
  -- (3) S 处理过的输入数据：上一时刻的预测值 - 此时刻的观测值
  -- Ss = trSTab[seqCnt]:clone()            -- 2维：mbSize * (M * M * 3) = mbSize * (opt.max_m * opt.in_size)
  local oneS = nil
  local begin = opt.in_size * (t-1) + 1
  oneS = Ss:narrow(2,begin,opt.in_size):clone()  -- 2维：mbSize * (1 * M * 3) = mbSize * (1 * opt.in_size) = mbSize * opt.in_size
  -- print(Ss) print(oneS) abort() 
  oneS = dataToGPU(oneS)
  table.insert(rnninp, oneS)


  return rnninp, rnn_state
end

--------------------------------------------------------------------------
--- RNN decoder
-- @param predictions	current predictions to use for feed back
-- @param t		time step (nil to predict for entire sequence)
function decodeDA(predictions, t)
  -- local loctimer=torch.Timer()
  local T = tabLen(predictions)	-- how many measurements

  local predDA = {}
  -- print('predictions') print(predictions) abort()
  if t ~= nil then
    local lst = predictions[t]
    if predDA then predDA = lst[opt.predDAIndex] end -- miniBatchSize x opt.nClasses
  else
    predDA = zeroTensor3(miniBatchSize,T,opt.nClasses)
    for tt=1,T do
      local lst = predictions[tt]
      predDA[{{},{tt},{}}] = lst[opt.predDAIndex]:clone():reshape(miniBatchSize, 1, opt.nClasses)
    end
  end

  return predDA
end