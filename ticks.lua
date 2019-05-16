-- perform one single RNN step (t to t+1)
function RNNTick(outputs, rnnSize, inputSizeL, x, prev_h, L)
  -- local rnnSize = opt.rnn_size
  -- real input to hidden 
  i2h = nn.Linear(inputSizeL, rnnSize)(x)

  -- hidden input to hidden
  local h2h = nn.Linear(rnnSize, rnnSize)(prev_h):annotate{name='h2h_DA_L'..L}
  
  -- RNN tick
  local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h}:annotate{name='Add DA'}):annotate{name='next_h_DA_L'..L}    

  table.insert(outputs, next_h)
  
  return outputs
end -- RNNTick

-- perform one single LSTM step (t to t+1)
function LSTMTick(outputs, rnnSize, inputSizeL, x, prev_h, prev_c, L)
  -- evaluate the input sums at once for efficiency
  -- 立即评估输入总和以提高效率
  local i2h = nn.Linear(inputSizeL, 4 * rnnSize)(x):annotate{name='i2h_'..L}
  local h2h = nn.Linear(rnnSize, 4 * rnnSize)(prev_h):annotate{name='h2h_'..L}
  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, rnnSize)(all_input_sums)
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  -- decode the gates
  local in_gate = nn.Sigmoid()(n1):annotate{name='in_gate_'..L,graphAttributes = {color='blue'}}
  local forget_gate = nn.Sigmoid()(n2):annotate{name='forget_gate_'..L,graphAttributes = {color='blue'}}
  local out_gate = nn.Sigmoid()(n3):annotate{name='out_gate_'..L,graphAttributes = {color='blue'}}
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4):annotate{name='in_transform_'..L,graphAttributes = {color='blue'}}
  
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
      })
      :annotate{name='next_c_'..L,
      graphAttributes = {color='red'}
    }
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
      :annotate{name='next_h_'..L,
      graphAttributes = {color='red'}
    }
  
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)
  
  return outputs
end -- LSTMTick

function getRNNMode(opt)
  
    local RNNMODE, LSTMMODE, GRUMODE = false,false,false
  if opt.model=='lstm' then LSTMMODE=true end
  if opt.model=='gru' then error('GRU not implemented') end
  if opt.model=='rnn' then RNNMODE=true end
  
  return RNNMODE, LSTMMODE, GRUMODE
end