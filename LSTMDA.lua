--[[
  An RNN that emulates a Bayesian Filter
]]--

require 'model.ticks';

local RNN = {}

function RNN.rnn(opt)
  local L = opt.num_layers  -- default 1
  local dropout = opt.dropout or 0
  local rnnSize = opt.rnn_size
  local inSize = opt.in_size
  local batchMode = opt.mini_batch_size>1

  local inputs = {}
  local outputs = {}

  -- (1) c
  table.insert(inputs, nn.Identity()():annotate{
    name='prev_c',
    description='prev. c',
    graphAttributes = {style='filled', fillcolor='red'}
  })
  -- (2) h
  table.insert(inputs, nn.Identity()():annotate{
    name='prev_h',
    description='prev. h',
    graphAttributes = {style='filled', fillcolor='red'}
  })

  -- (3) S t时刻：一个目标的预测值，训练的时候用真实值 减去 t时刻：所有的测量值，含杂波
  table.insert(inputs, nn.Identity()():annotate{
    name='x_i_t -Z _all_t',
    description="one target's state X",
    graphAttributes = {style='filled', fillcolor='red'}
  })

 
  ---------------------------------------------------------
  -------------------DA  PREDICTION---------------------
  ---------------------------------------------------------
  -- L>1 has not been implemented
  local pc = inputs[opt.pcIndex]
  local ph = inputs[opt.phIndex]
  local s = inputs[opt.sIndex]  -- size: batch * inSize

  -- local lstminp = nn.JoinTable(1,1){x, z} 

  local DA_state = {}
  DA_state = LSTMTick(DA_state, rnnSize, inSize, s, ph, pc, 1)
  -- function LSTMTick(outputs, nHidden, inputSizeL, x, prev_h, prev_c, L)

  local top_DA_state = DA_state[#DA_state] -- 取h

  if dropout > 0 then top_DA_state = nn.Dropout(dropout)(top_DA_state) end
  local da = nn.Linear(rnnSize, opt.max_m)(top_DA_state):annotate{
    name='DA_t',
    description='data assoc.',
    graphAttributes = {color='green', style='filled'}
  }

  local localDaRes = nn.Reshape(opt.max_m, batchMode)(da):annotate{name='Rshp DA'}

  local daFinal = localDaRes

  -- daFinal = nn.LogSoftMax()(localDaRes):annotate{
  --   name='DA_t',
  --   description='data assoc. LogSoftMax',
  --   graphAttributes = {color='green'}
  -- }
  
  daFinal = nn.Sigmoid()(localDaRes):annotate{
    name='DA_t',
    description='data assoc. LogSoftMax',
    graphAttributes = {color='green'}
  }
  ---------------------------------------------------------
  --------------------------output-------------------------
  ---------------------------------------------------------
  for _,v in pairs(DA_state) do table.insert(outputs, v) end
  -- table.insert(outputs, next_c)
  -- table.insert(outputs, next_h)
  table.insert(outputs, daFinal)
  
  return nn.gModule(inputs, outputs)--最后使用nn.gModule({input},{output})来定义神经网络模块。 

end
return RNN

