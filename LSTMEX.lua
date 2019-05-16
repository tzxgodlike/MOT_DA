--[[
  An RNN that emulates a Bayesian Filter
]]--

require 'model.ticks';

local RNN = {}

function RNN.rnn(opt)
  local L = opt.num_layers  -- default 1
  local dropout = opt.dropout or 0
  local rnnSize = opt.rnn_size
  local batchMode = opt.mini_batch_size>1

  local inputs = {}
  local outputs = {}

  table.insert(inputs, nn.Identity()():annotate{
    name='prev_h',
    description='prev. h',
    graphAttributes = {style='filled', fillcolor='red'}
  })

  table.insert(inputs, nn.Identity()():annotate{
    name='A_t',
    description='prev. state X',
    graphAttributes = {style='filled', fillcolor='red'}
  })

  
  table.insert(inputs, nn.Identity()():annotate{
    name='Ex_t',
    description='detect state Z',
    graphAttributes = {style='filled', fillcolor='red'}
  })


 
  ---------------------------------------------------------
  -------------------EX     PREDICTION---------------------
  ---------------------------------------------------------
  -- L>1 has not been implemented
  local ph = inputs[opt.phIndex]
  local a = inputs[opt.aIndex]
  local ex = inputs[opt.exIndex]

  local a2i = nn.Linear(opt.max_n*nClasses, rnnSize)(a)
  local ex2i = nn.Linear(opt.max_n, rnnSize)(ex)

  local rnninp = nn.JoinTable(1,1){a2i, ex2i}
  rnninp = nn.Reshape(2*rnnSize, true){rnninp}

  local i2h = nn.Linear(2*rnnSize, rnnSize)(rnninp)
  local h2h = nn.Linear(rnnSize, rnnSize)(ph)
  local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})
  -- local next_h = nn.Sigmoid()(nn.CAddTable(){i2h, h2h})
  -- local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h})

  local ExPred =  nn.Linear(rnnSize, opt.max_n)(next_h)


  ---------------------------------------------------------
  --------------------------output-------------------------
  ---------------------------------------------------------
  table.insert(outputs, next_h)
  table.insert(outputs, ExPred)
  
  return nn.gModule(inputs, outputs)--最后使用nn.gModule({input},{output})来定义神经网络模块。 

end
return RNN

