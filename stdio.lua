--------------------------------------------------------------------------
--- Prints column names for process table
function printTrainingHeadline()
  local headline = 
    string.format("%14s%10s%8s%9s%8s%6s%6s","Iter.","Tr. loss","G-norm","tm/btch","l-rate","ETL","TELP")
  print(headline)
end

--------------------------------------------------------------------------
--- Prints numbers for current iteration
-- @param sec 	Seconds passed since start
function printTrainingStats(i, me, tl, gn, t, lr, sec)
  local secLeft = (sec / (i/opt.max_epochs) - sec)
  local hrsLeft = math.floor(secLeft / 3600)
  local minLeft = torch.round((secLeft % 3600)/60)
  local hrsElapsed = math.floor(sec / 3600)
  local minElapsed = torch.round((sec % 3600)/60)
  
  print(string.format("%6d/%7d%10.7f %.1e%8.2fs %.1e%3d:%02d%3d:%02d", i, me, tl, gn, t,lr, hrsLeft,minLeft,hrsElapsed,minElapsed))
end


function plotTraingDAProcess( predDA, DAgts, T )
  -- print('T = '..T)
  local firstm = string.format('   %1s %1s %1s %1s %1s |%8s %8s %8s %8s %8s','m1','m2','m3','m4','m5','pred1','pred2','pred3','pred4','pred5')
  print(firstm)
  local left, right = '', ''
  local gt = DAgts:narrow(1,1,1):clone():reshape(5,5) 
  -- print(predDA)
  for i = 1, T do
    local pDA = predDA[i]:narrow(1,1,1):clone() -- 1 * M+1
    -- print(pDA)
    left = string.format('n%d%2.0f %2.0f %2.0f %2.0f %2.0f |',i,gt[i][1],gt[i][2],gt[i][3],gt[i][4],gt[i][5])
    right = string.format(' %1.5f %1.5f %1.5f %1.5f %1.5f',pDA[1][1],pDA[1][2],pDA[1][3],pDA[1][4],pDA[1][5])
    print(left..right)
    
  end
  -- abort()

end

--------------------------------------------------------------------------
--- Model-specific options in a line
function printModelOptions(opt)
  local modelParams = opt.modelParams
  local header = ''
  local params = ''
  for k,v in pairs(modelParams) do 
    --若模型参数字符串长度大于5，只取前五个字符
    if string.len(v) > 5 then v = string.sub(v,1,5) end
    header = header..string.format('%6s',v) 
  end
  for k,v in pairs(modelParams) do params = params..string.format('%6d',opt[v]) end
  print(header)
  print(params)
end

function printDim(data, dim)
  dim=dim or 1
  local N,F,D = getDataSize(data)
  if opt.mini_batch_size == 1 then
    print(data:narrow(3,dim,1):reshape(N,F))
  else
    N = N / opt.mini_batch_size
    for mb = 1,opt.mini_batch_size do
      local mbStart = N * (mb-1)+1
      local mbEnd =   N * mb
      local data_batch = data[{{mbStart, mbEnd}}]:clone()
      print(data_batch:narrow(3,dim,1):reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end
end

function printAll(tr, det, lab, ex, detex)
  local N,F,D = getDataSize(tr)
  
  if lab=={} then lab=nil end
  if ex=={} then lab=nil end
  if detex=={} then lab=nil end
  
  local dim = 1
  print('--------   Tracks   -----------')
  printDim(tr, dim)
  
  print('-------- Detections -----------')
  printDim(det, dim)
  
  
  local N = lab:size(1)/opt.mini_batch_size
  if lab~= nil then 
  print('--------   Labels   -----------')
  -- print(lab)  
  for mb = 1,opt.mini_batch_size do
    local mbStart = opt.max_n * (mb-1)+1
    local mbEnd =   opt.max_n * mb
    local data_batch = lab[{{mbStart, mbEnd}}]:clone()
    -- print(data_batch)
    -- print(N,F)
    print(data_batch:reshape(N,F))
    if mb < opt.mini_batch_size then print('---') end
  end     
  end
    
  
  if ex~= nil then 
    local N = ex:size(1)/opt.mini_batch_size
    print('--------  Existance -----------')    
    for mb = 1,opt.mini_batch_size do
      local mbStart = opt.max_n * (mb-1)+1
      local mbEnd =   opt.max_n * mb
      local data_batch = ex[{{mbStart, mbEnd}}]:clone()
    -- print(data_batch)
    -- print(N,F)      
      print(data_batch:reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end
    
  if detex~= nil then
    local N = detex:size(1)/opt.mini_batch_size
    print('------- Det. Existance --------')
    for mb = 1,opt.mini_batch_size do
      local mbStart = N * (mb-1)+1
      local mbEnd =   N * mb
      local data_batch = detex[{{mbStart, mbEnd}}]:clone()
      -- print(data_batch)
      -- print(N,F)
      print(data_batch:reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end

  
  print('-------------------------------')
end

function printDetEx(detex)
  if detex~= nil then
    local N = detex:size(1)/opt.mini_batch_size
    print('------- Det. Existance --------')
    for mb = 1,opt.mini_batch_size do
      local mbStart = N * (mb-1)+1
      local mbEnd =   N * mb
      local data_batch = detex[{{mbStart, mbEnd}}]:clone()
      -- print(data_batch)
      print(N,F)
      print(data_batch:reshape(N,F))
      if mb < opt.mini_batch_size then print('---') end
    end    
  end
end
