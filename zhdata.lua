--pre processing data

--- The different between data processing:
--- KF: aiming to train a one-to-one tracking model, the tracksTab and the detsTab are with the same size: nxFxD(n=1)
--- DA: aiming to train a one-to-multi data association model, tracksTab's size is nxFxD, detsTab'size is mxFxD.

function getDataPath()
  -- local dataDir = '../data/multi/'
  local dataDir=''
  if TRAINING_BF or TESTING_BF then
    if TRAINING_BF100 then
      dataDir = lfs.currentdir()..'/../data/trainBF100/'
    elseif TRAINING_BF300 then
      dataDir = lfs.currentdir()..'/../data/trainBF300/'
    elseif TRAINING_BF450 then
      dataDir = lfs.currentdir()..'/../data/trainBF450/'
    elseif TESTING_BF100 then
      dataDir = lfs.currentdir()..'/../data/testBF100/'
    elseif TESTING_BF300 then
      dataDir = lfs.currentdir()..'/../data/testBF300/'
    elseif TESTING_BF450 then
      dataDir = lfs.currentdir()..'/../data/testBF450/'
    end
  end
  if TRAINING_DA then
    -- if TRAINING_DA100 then
      dataDir = lfs.currentdir()..'/../data/trHA_MT_N5_M5/'
    -- elseif TRAINING_DA300 then
    --   dataDir = lfs.currentdir()..'/../data/trMT300/'
    -- elseif TRAINING_DA450 then
    --   dataDir = lfs.currentdir()..'/../data/trMT450/'
    -- end
  end
  -- print(dataDir)
  -- make sure directory exists
  assert(lfs.attributes(dataDir, 'mode'), 'No data dir found')
  
  return dataDir     
end

function preBFData(mode, sequences)
  -- body
  print('Preparing '..mode..' data...')

  local TracksTab, DetsTab ={}, {}

  for _, seqName in pairs(sequences) do
    -- 1. get the raw data Tensor

    local tracks = getGtTracks(seqName)

    local maxtrackPerFrame, Fs = tracks:size(1), tracks:size(2)
    local detections = getDets(seqName, Fs)
    -- local detections = getDets(seqName)


    -- 2. get the max state value of tracks and detections
    GetMaxMinState(tracks, detections, seqName)

    -- 3. normalization min-max scaling
    local Ngt, Fgt, Dgt = getDataSize(tracks)
    local maxX, minX = maxmin_State[seqName]['MaxX'], maxmin_State[seqName]['MinX']
    local maxY, minY = maxmin_State[seqName]['MaxY'], maxmin_State[seqName]['MinY']
    local maxZ, minZ = maxmin_State[seqName]['MaxZ'], maxmin_State[seqName]['MinZ']
    if opt.norm_type == 1 then
      tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - minX)/(maxX - minX)
      detections[{{},{},{1}}] = (detections[{{},{},{1}}] - minX)/(maxX - minX)
      tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - minY)/(maxY - minY)
      detections[{{},{},{2}}] = (detections[{{},{},{2}}] - minY)/(maxY - minY)
      tracks[{{},{},{3}}] = (tracks[{{},{},{3}}] - minZ)/(maxZ - minZ)
      detections[{{},{},{3}}] = (detections[{{},{},{3}}] - minZ)/(maxZ - minZ)
    elseif opt.norm_type == 2 then
      tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - maxX/2)/maxX
      detections[{{},{},{1}}] = (detections[{{},{},{1}}] - maxX/2)/maxX
      tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - maxY/2)/maxY
      detections[{{},{},{2}}] = (detections[{{},{},{2}}] - maxY/2)/maxY
      tracks[{{},{},{3}}] = (tracks[{{},{},{3}}] - maxZ/2)/maxZ
      detections[{{},{},{3}}] = (detections[{{},{},{3}}] - maxZ/2)/maxZ
    end
    -- print('tracks') print(tracks)
    -- print('detections') print(detections)
    -- abort()
    table.insert(TracksTab,tracks)  --(seqNum, N, F, D)
    table.insert(DetsTab,detections)  --(seqNum, N, F, D)

  end
  -- print('TracksTab') print(TracksTab[10])
  -- print('DetsTab') print(DetsTab[10])
  -- abort()
  local nProc =0
  if miniBatchSize>1 then nProc = opt.data_train else nProc = opt.data_valid end
  
  local TRACKsTab, DETsTab = processBFData(#sequences, TracksTab ,DetsTab,nProc, miniBatchSize)

  return TRACKsTab, DETsTab
end

--------------------------------------------------------------------------
--- preprocess one-to-one ,gt-to-det data
function processBFData(seqNum, tracksTab ,detsTab, nProc, mbSize)
  print('nProc'..nProc)
  local trTracksTab, trDetsTab = {}, {}
  local tmpwin = opt.temp_win
  for n=1,nProc do
    local alltr = torch.zeros(1, tmpwin, stateDim)
    local alldet = torch.zeros(1, tmpwin, stateDim)
    for m=1,mbSize do
      local randSeq= math.random(seqNum) -- select a sequence randomly
      local atrack = tracksTab[randSeq]
      local adet = detsTab[randSeq]
      local N,F,D = getDataSize(atrack) -- print(N,F,D)
      
      local randTar = math.random(N)  -- select a target randomly form a sequence

      local tr = torch.zeros(1,tmpwin, opt.state_dim)
      local det = torch.zeros(1,tmpwin, opt.state_dim)

      local trajS = math.random(F-tmpwin)
      local trajE = trajS + tmpwin - 1
      -- print('randSep:'..randSeq..', trajS..'..trajS..', trajE..'..trajE)

      tr = atrack[{{randTar},{trajS,trajE}}]
      det = adet[{{randTar},{trajS,trajE}}]
      -- print('tr') print(tr)
      -- print('det') print(det)
      -- abort()
      alltr = alltr:cat(tr, 1)
      alldet = alldet:cat(det, 1)

    end
    alltr = alltr:sub(2,-1)
    alldet = alldet:sub(2,-1)
    -- print('alltr') print(alltr) abort()
    table.insert(trTracksTab, alltr)
    table.insert(trDetsTab, alldet)

  end
  -- abort()
  -- print('process')
  -- print('trTracksTab') print(trTracksTab)
  -- print('trDetsTab') print(trDetsTab)
  return trTracksTab, trDetsTab
end

function preDAData(mode, sequences)
  -- body
  print('Preparing DA '..mode..' data...')

  local DAxTab, DAzTab, DAgtTab ={}, {}, {}
  Fs = torch.zeros(#sequences)  --global var
  for k, seqName in pairs(sequences) do
    local fn = getDataPath() .. seqName .. '.mat'
    local loaded = matio.load(fn) 
    -- print(loaded) abort()
    if loaded.allDAz:size(1) ~= opt.max_m then
      print('The number of measurements for a time step is not equal opt.max_m!!!!')
      abort()
    end
      -- allDAgt : DoubleTensor - size: 5x6x549            use
      -- Appear_disappearT : DoubleTensor - size: 3x2      3 means the num of real target
      -- allMeas : DoubleTensor - size: 5x3x550            no use for DA
      -- allStates : DoubleTensor - size: 3x3x550          no use for DA
      -- Meas_Tar : DoubleTensor - size: 550x5             no use for DA
      -- allDAx : DoubleTensor - size: 5x3x549             use
      -- allDAz : DoubleTensor - size: 5x3x549             use
      -- maxF : DoubleTensor - size: 1x1                   use
    Fs[k] = loaded.maxF
    local allDAx, allDAz, allDAgt = loaded.allDAx:float(), loaded.allDAz:float(), loaded.allDAgt:float()
    -- print('allDAx') print(allDAx[{{},{},{1}}])  
    -- print('allDAz') print(allDAz[{{},{},{1}}])

    -- 2. get the max state value of tracks and detections
    maxState, minState = GetMaxMinState2(allDAx, allDAz, seqName)
    -- print(maxState) print(minState) print(maxState[1]) abort()
    
    -- 3. 根据参数选择标准化或归一化，normalization min-max scaling
    if opt.norm_type == 1 then
      allDAx[{{},{1},{}}] = (allDAx[{{},{1},{}}] - minState[1]) / (maxState[1] - minState[1])  -- x
      allDAx[{{},{2},{}}] = (allDAx[{{},{2},{}}] - minState[2]) / (maxState[2] - minState[2])  -- y
      -- allDAx[{{},{3},{}}] = (allDAx[{{},{3},{}}] - minState[3]) / (maxState[3] - minState[3])  -- z

      allDAz[{{},{1},{}}] = (allDAz[{{},{1},{}}] - minState[1]) / (maxState[1] - minState[1])  -- x
      allDAz[{{},{2},{}}] = (allDAz[{{},{2},{}}] - minState[2]) / (maxState[2] - minState[2])  -- y
      -- allDAz[{{},{3},{}}] = (allDAz[{{},{3},{}}] - minState[3]) / (maxState[3] - minState[3])  -- z
    elseif opt.norm_type == 2 then
      allDAx[{{},{1},{}}] = (allDAx[{{},{1},{}}] - maxState[1]/2) / maxState[1]   -- x
      allDAx[{{},{2},{}}] = (allDAx[{{},{2},{}}] - maxState[2]/2) / maxState[2]   -- y
      -- allDAx[{{},{3},{}}] = (allDAx[{{},{3},{}}] - maxState[3]/2) / maxState[3]   -- z

      allDAz[{{},{1},{}}] = (allDAz[{{},{1},{}}] - maxState[1]/2) / maxState[1]   -- x
      allDAz[{{},{2},{}}] = (allDAz[{{},{2},{}}] - maxState[2]/2) / maxState[2]   -- y
      -- allDAz[{{},{3},{}}] = (allDAz[{{},{3},{}}] - maxState[3]/2) / maxState[3]   -- z
    end
    -- print('allDAx') print(allDAx[{{},{},{1}}])  
    -- print('allDAz') print(allDAz[{{},{},{1}}])
    -- abort()
    table.insert(DAxTab,allDAx)  --(seqNum, N, D, F)
    table.insert(DAzTab,allDAz)  --(seqNum, N, D, F)
    table.insert(DAgtTab,allDAgt)  --(seqNum, N, D, F)
  end
  -- print('DAxTab') print(DAxTab)
  -- print('DAzTab') print(DAzTab)
  -- print(Fs)
  -- abort()
  local nProc =0
  if miniBatchSize>1 then nProc = opt.data_train else nProc = opt.data_valid end
  
  local trSTab, trDAgtTab = processDAData(#sequences, DAxTab ,DAzTab,DAgtTab,nProc, miniBatchSize)

  return trSTab, trDAgtTab
end

--------------------------------------------------------------------------
--- preprocess one-to-one ,gt-to-det data
function processDAData(seqNum, DAxTab ,DAzTab, DAgtTab, nProc, mbSize)
  print('nProc'..nProc)
  local allSsize, gtsize= opt.max_m*opt.in_size, opt.max_m * (opt.max_m)
  local trSTab, trDAgtTab = {}, {}
  for n = 1, nProc do
    local batchS = torch.Tensor(1, allSsize):fill(0) --一个minibatch的DA输入数据
    local batchDAgt = torch.Tensor(1, gtsize):fill(0) --一个minibatch的DA输入数据
    for m = 1, mbSize do
      local randSeq = math.random(seqNum)                   -- 随机选取一个序列
      local randF = math.random(Fs[randSeq])                -- 在从这个序列中随机选取一帧
      local dax = DAxTab[randSeq][{{},{},{randF}}]:clone():squeeze()    -- M * D 
      local daz = DAzTab[randSeq][{{},{},{randF}}]:clone():squeeze()    -- M * D 
      local dagt = DAgtTab[randSeq][{{},{},{randF}}]:clone():squeeze()  -- M * M+1 
      -- print(dax) print(daz) print(dagt) abort()
      dagt = torch.reshape(dagt, 1, gtsize)
      -- print(dax) print(daz) print(dagt)
      local oneS = torch.Tensor(1, opt.in_size):fill(0)  -- 处理后的DA网络的输入，维度为 1 * (M*M*D)
      for i = 1, opt.max_m do
        local s = torch.repeatTensor(dax[{{i}}]:clone():squeeze(), opt.max_m , 1) -- 将一个x重复max_m次
        -- print('init s:') print(s)
        s = s - daz   -- sub
        -- print('sub s') print(s) print(oneS) abort()
        s = torch.reshape(s,1,opt.in_size)
        oneS = oneS:cat(s,1)
      end
      oneS = oneS:sub(2,-1)  --去掉第一行的全0
      oneS = torch.reshape(oneS,1,allSsize)
      -- print(oneS) print(dagt) 

      batchS = batchS:cat(oneS,1)
      batchDAgt = batchDAgt:cat(dagt,1)
    end
    batchS = batchS:sub(2,-1)       -- mbSize * (M * M * 3)
    batchDAgt = batchDAgt:sub(2,-1) -- mbSize * (M * (M+1))
    -- print(batchS) print(batchDAgt) abort()
    table.insert(trSTab, batchS)       -- data_train: mbSize * (M * M * 3)
    table.insert(trDAgtTab, batchDAgt) -- data_train: mbSize * (M * (M+1))
  end
  -- abort()
  return trSTab, trDAgtTab
end
-------------------------------------------------------------------------
--- Return the max and min state of tracks an detections
--- Note: the detections can not be nil!!!!!
function GetMaxMinState(tracks, detections, seqName)
  maxmin_State = {}  -- global var
  local max=detections[1][1]:resize(1,3):clone()
  local min=max:clone()
  if maxmin_State[seqName] == nil then maxmin_State[seqName] = {}
    ---gt中状态的最大值---------
    if tracks~=nil then
      local Ngt, Fgt, Dgt = getDataSize(tracks)
      for i=1,Ngt do
        -- print('n='..i)
        max=max:cat(torch.max(tracks[i],1),1)
        min=min:cat(torch.min(tracks[i],1),1)
        -- print(max) print(min)
        -- abort()
      end
    end
    ---det中状态的最大值--------
    local Ndet,Fdet,Det=getDataSize(detections)
    for i=1,Ndet do
      -- print('m='..i)
      max=max:cat(torch.max(detections[i],1),1)
      min=min:cat(torch.min(detections[i],1),1)
      -- print(max) print(min)
    end
    -- gt和det中状态的最大值
    -- min = min:sub(3,-1)
    -- min = min:narrow(1,2,2)  -- remove the first zero row
    max=torch.max(max,1)
    min=torch.min(min,1)
    -- print(max) print(min)
    -- abort()
    maxmin_State[seqName]['MaxX'], maxmin_State[seqName]['MaxY'], maxmin_State[seqName]['MaxZ']=max[1][1],max[1][2],max[1][3]
    maxmin_State[seqName]['MinX'], maxmin_State[seqName]['MinY'], maxmin_State[seqName]['MinZ']=min[1][1]-1,min[1][2]-1,min[1][3]-1
    -- print(maxmin_State[seqName]['MaxX'])
    -- print(maxmin_State[seqName]['MaxY'])
    -- print(maxmin_State[seqName]['MinX'])
    -- print(maxmin_State[seqName]['MinY'])
    -- print(maxmin_State[seqName]['MaxZ'])
    -- print(maxmin_State[seqName]['MinZ'])
    -- abort()
  end
end

-------------------------------------------------------------------------
--- Return the max and min state of tracks an detections
--- Note: the detections can not be nil!!!!!
function GetMaxMinState2(dax, daz, seqName)
  maxmin_State = {}  -- global var
  maxState, minState = {},{}
  if maxmin_State[seqName] == nil then maxmin_State[seqName] = {}
    local maxDAx = torch.max(dax,3) -- 每个检测中的所有帧中最大值
    maxDAx = torch.max(maxDAx,1) -- 再找最大值
    local maxDAz = torch.max(daz,3) -- 每个检测中的所有帧中最大值
    maxDAz = torch.max(maxDAz,1) -- 再找最大值
    maxState = maxDAx:cat(maxDAz, 1)
    maxState = torch.max(maxState,1) :squeeze()

    local minDAx = torch.min(dax,3) -- 每个检测中的所有帧中最大值
    minDAx = torch.min(minDAx,1) -- 再找最大值
    local minDAz = torch.min(daz,3) -- 每个检测中的所有帧中最大值
    minDAz = torch.min(minDAz,1) -- 再找最大值
    minState = minDAx:cat(minDAz, 1)
    minState = torch.min(minState,1) :squeeze()

    maxmin_State[seqName]['Max'] = maxState
    maxmin_State[seqName]['Min'] = minState - 1
    -- print(minState) print(maxmin_State[seqName]['Max'])  print(maxmin_State[seqName]['Min'])  abort()
  end
  return maxState, minState
end

------------------------------------------------------------------
----  return the normalized 'gt' and 'det' data for testing.
----  tracks: 1 * F * D  tensor
----  detections: 1 * F * D tensor
function testBFData(mode, aSeqName)
  -- body
  print('Preparing '..mode..' data...')

  local rawtracks = getGtTracks(aSeqName)

  local maxtrackPerFrame, Fs = rawtracks:size(1), rawtracks:size(2)
  local rawdetections = getDets(aSeqName, Fs)

  local tracks=rawtracks:clone()
  local detections = rawdetections:clone()

  -- 2. get the max state value of tracks and detections
  GetMaxMinState(tracks, detections, aSeqName)

  -- 3. normalization min-max scaling
  local Ngt, Fgt, Dgt = getDataSize(tracks)
  local maxX, minX = maxmin_State[aSeqName]['MaxX'], maxmin_State[aSeqName]['MinX']
  local maxY, minY = maxmin_State[aSeqName]['MaxY'], maxmin_State[aSeqName]['MinY']
  local maxZ, minZ = maxmin_State[aSeqName]['MaxZ'], maxmin_State[aSeqName]['MinZ']
  if opt.norm_type == 1 then
    tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - minX)/(maxX - minX)
    detections[{{},{},{1}}] = (detections[{{},{},{1}}] - minX)/(maxX - minX)
    tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - minY)/(maxY - minY)
    detections[{{},{},{2}}] = (detections[{{},{},{2}}] - minY)/(maxY - minY)
    tracks[{{},{},{3}}] = (tracks[{{},{},{3}}] - minZ)/(maxZ - minZ)
    detections[{{},{},{3}}] = (detections[{{},{},{3}}] - minZ)/(maxZ - minZ)
  elseif opt.norm_type == 2 then
    tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - maxX/2)/maxX
    detections[{{},{},{1}}] = (detections[{{},{},{1}}] - maxX/2)/maxX
    tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - maxY/2)/maxY
    detections[{{},{},{2}}] = (detections[{{},{},{2}}] - maxY/2)/maxY
    tracks[{{},{},{3}}] = (tracks[{{},{},{3}}] - maxZ/2)/maxZ
    detections[{{},{},{3}}] = (detections[{{},{},{3}}] - maxZ/2)/maxZ
  end
    -- print('tracks') print(tracks)
    -- print('detections') print(detections)
    -- abort()
  return rawtracks, rawdetections, tracks, detections
end


--------------------------------------------------------------------------
--- tracks: N x F x D Tensor
function getGtTracks(seqName)
  local nDim = opt.state_dim  -- number of dimensions
  -----load gt------
  local gtfile = getDataPath() .. seqName .. "/gt.txt"
  local gt = 0
  if lfs.attributes(gtfile) then 
    print(string.format("   get sequence %s's GT file.",seqName))
    gt = readTXTzh(gtfile, 1) -- param #2 = GT
  else
    error("ERROR : GT file ".. gtfile .." does not exist")
  end
  

  local _,_, maxKey = tabLen(gt)--Returns number of elements in a table, as well as min. and max. key
  local F = maxKey
  local N = 10--max. number of targets / detections

  local tracks = torch.ones(N, F, nDim):fill(0)

  -- Now create a FxN tensor with states
  for t in pairs(gt) do
    local cnt=0
    for i = 1, N do
      cnt=cnt+1
      local gtBBox = gt[t][i]--某一帧t中第i个目标数据，为1X3维的

      if gtBBox then 
        for dim = 1,nDim do     
         tracks[cnt][t][dim]=gtBBox[1][dim]
        end

      end

    end
  end

  -- 清楚所有零行张量
  tracks = cleanDataTensor(tracks)
  return tracks
end

--------------------------------------------------------------------------
---detections: M x F x D Tensor
-- function getDets(seqName, tracks)
function getDets(seqName, Frames)
  local nDim = opt.state_dim  -- number of dimensions
  local detfile = getDataPath() .. seqName .. "/det.txt"
  if lfs.attributes(detfile) then 
    print(string.format("   get sequence %s's DET file.",seqName))
  else
    error("Error: Detections file ".. detfile .." does not exist")
  end
  local det = readTXTzh(detfile, 2)

  local maxDetPerFrame = 0
  for k,v in pairs(det) do 
    -- if k>F then F=k end 
    if tabLen(v)>maxDetPerFrame then maxDetPerFrame = tabLen(v) end
  end
  --（每一帧最大测量数，帧数，状态维数）
  local detections = torch.zeros(maxDetPerFrame,Frames, nDim)
  
  for t=1,Frames do
    if det[t] then -- skip if no detections present in frame 
      for detID,detBBox in pairs(det[t]) do      
          detections[{{detID},{t},{}}] = detBBox:narrow(2,1,nDim)
      end
    end
  end

  -- local Ndet,Fdet,Ddet = getDataSize(detections)
  -- if Ndet<1 then detections=torch.zeros(1,F,nDim); maxDetPerFrame=1 end

  -- -- pad detections with empty frames at end if necessary
  -- --真值的帧数大于测量值的帧数时，在末端填充空帧,
  -- local Ngt,Fgt,Dgt = getDataSize(tracks)
  -- if Fgt>F then
  --   detections=detections:cat(torch.zeros(maxDetPerFrame,Fgt-F,nDim),2)
  -- end
  -- print('detections') print(detections)
  -- abort()
  return detections
end



function preTestData(mode, sequences)
  print('Preparing '..mode..' data...')

  maxmin_State = {}  -- global var

  local TracksTab, DetsTab ={}, {}

  for _, seqName in pairs(sequences) do
    -- 1. get the raw data Tensor
    local tracks = getGtTracks(seqName, 'TEST')
    local detections = getDets(seqName, tracks, 'TEST')

    -- 2. get the max state value of tracks and detections
    GetMaxMinState(tracks, detections, seqName)

    -- 3. normalization min-max scaling
    local Ngt, Fgt, Dgt = getDataSize(tracks)
    local maxX, minX = maxmin_State[seqName]['MaxX'], maxmin_State[seqName]['MinX']
    local maxY, minY = maxmin_State[seqName]['MaxY'], maxmin_State[seqName]['MinY']
    
    tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - minX)/(maxX - minX)
    detections[{{},{},{1}}] = (detections[{{},{},{1}}] - minX)/(maxX - minX)
    tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - minY)/(maxY - minY)
    detections[{{},{},{2}}] = (detections[{{},{},{2}}] - minY)/(maxY - minY)
    -- print('tracks') print(tracks)
    -- print('detections') print(detections)
    table.insert(TracksTab,tracks)  --(seqNum, N, F, D)
    table.insert(DetsTab,detections)  --(seqNum, N, F, D)

  end

  return TracksTab, DetsTab
end

--------------------------------------------------------------
--------------
function preAllData(mode, sequences)
  print('Preparing '..mode..' All data...')

  maxmin_State = {}  -- global var

  local TracksTab, DetsTab, DasTab ={}, {}, {}

  for _, seqName in pairs(sequences) do
    -- -- 1. get the raw data Tensor
    local tracks = getGtTracks(seqName)               -- nxFxD, n=1
    local maxtrackPerFrame, Fs = tracks:size(1), tracks:size(2)
    local detections = getDets(seqName, Fs)           -- mxFxD
    local maxDetPerFrame =detections:size(1)
    local das = getDAs(seqName, maxtrackPerFrame, Fs) -- nxFx(m+1), n=1
    -- print(das) abort()
    
    -- 2. get the max state value of tracks and detections
    GetMaxMinState(tracks, detections, seqName)

    -- 3. normalization min-max scaling
    local Ngt, Fgt, Dgt = (tracks)
    local maxX, minX = maxmin_State[seqName]['MaxX'], maxmin_State[seqName]['MinX']
    local maxY, minY = maxmin_State[seqName]['MaxY'], maxmin_State[seqName]['MinY']

    tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - maxX/2)/maxX
    detections[{{},{},{1}}] = (detections[{{},{},{1}}] - maxX/2)/maxX
    tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - maxY/2)/maxY
    detections[{{},{},{2}}] = (detections[{{},{},{2}}] - maxY/2)/maxY

    -- tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - minX)/(maxX - minX)
    -- detections[{{},{},{1}}] = (detections[{{},{},{1}}] - minX)/(maxX - minX)
    -- tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - minY)/(maxY - minY)
    -- detections[{{},{},{2}}] = (detections[{{},{},{2}}] - minY)/(maxY - minY)

    -- print('tracks') print(tracks)
    -- print('detections') print(detections) abort()
    table.insert(TracksTab,tracks)    --(seqNum, N, F, D)
    table.insert(DetsTab,detections)  --(seqNum, M, F, D)
    table.insert(DasTab,das)          --(seqNum, N, F, M+1)
  end
  local nProc = 1
  if miniBatchSize>1 then nProc = opt.data_train else nProc = opt.data_valid end
  -- print('nProc='..nProc)
  local TRACKsTab, DETsTab, DAsTab, EXsTab, DetEXsTab, SeqNamesTab = processAllData(sequences, TracksTab ,DetsTab, DasTab, nProc, miniBatchSize)
  return TRACKsTab, DETsTab, DAsTab, EXsTab, DetEXsTab, SeqNamesTab
end

--------------------------------------------------------------------------
--- preprocess one(target)-to-multi(detections) data
function processAllData(sequences, tracksTab ,detsTab, dasTab, nProc, mbSize)
  local trTracksTab, trDetsTab, trDasTab, trExTab, trDetExTab, trSeqNameTab= {}, {}, {}, {}, {}, {}
  local seqNum = #sequences
  local tmpwin = opt.temp_win
  local birth_death = true
  for n=1,nProc do
    local alltr = torch.zeros(1, tmpwin, stateDim)
    local alldet = torch.zeros(1, tmpwin, stateDim)
    local allda = torch.zeros(1, tmpwin, nClasses)
    local allex = torch.zeros(1, tmpwin)
    local alldetex = torch.zeros(1,tmpwin, maxDets)
    local allseqnames = {}
    for m=1,mbSize do
      local randSeq= math.random(seqNum) -- select a sequence randomly
      local seqName = sequences[randSeq]
      local alltrack = tracksTab[randSeq]                -- N x F x D
      local dets = detsTab[randSeq]                      -- M x F x D
      local das = dasTab[randSeq]                        -- N x F x M+1
  
      local N= alltrack:size(1)
      local randTar = math.random(N)     -- select a target randomly form a sequence
      -- randTar=1
      -- local atrack = alltrack[{{randTar}}]               -- 1 x F x D
      -- local ada = allda[{{randTar}}]                     -- 1 x F x M+1
    
      local cur_Tar_len = das[{{randTar},{},{nClasses}}]:eq(0):squeeze()
      local F = cur_Tar_len:sum()
      -- print(randTar,F) 

      local trajS = math.random(F-tmpwin)
      local trajE = trajS + tmpwin - 1
      -- print('trajS..'..trajS..',trajE..'..trajE)

      -- deep copy!!!!!!!!!!!!
      local atrack = alltrack[{{randTar},{trajS,trajE}}]:clone() -- 1 x F(tmpwin) x D
      -- print('atrack') print(atrack) abort()
      local dets = dets[{{},{trajS,trajE}}]:clone()              -- M x F(tmpwin) x D
      -- print('dets ') print(dets) 
      local ada = das[{{randTar},{trajS,trajE}}]:clone()         -- 1 x F(tmpwin) x M+1
      -- print('ada') print(ada) print(randTar) abort()
      local ex = torch.ones(1, tmpwin)
      local detex = torch.ones(1,tmpwin, maxDets)

      -- initialize ex
      for t = 1, tmpwin do
        if ada[1][t][nClasses] == 1 then
          ex[1][t] = 0
        end
      end

      -- birth/death : fix ex
      local d_rate = torch.rand(1)
      -- print(d_rate)
      local exS = math.random(tmpwin/4)
      local exE = tmpwin - torch.random(tmpwin-3)+1
      if d_rate[1]<opt.death_rate and TRAINING then
        if exS>1 then ex[{{1},{1,exS-1}}]=0 end
        if exE-exS>2 and exE<tmpwin then
          ex[{{1},{exE+1,tmpwin}}]=0
        end
      end
      -- fix da/det based on ex
      local dRange = torch.zeros(3,stateDim)
      for d=1,stateDim do
        dRange[1][d]=-.5
        dRange[2][d]=.5 
        dRange[3][d]=1
      end

      for t =1, tmpwin do
        if ex[1][t]==0 then
          ada[1][t]=torch.zeros(1,nClasses)
          ada[1][t][nClasses] = 1
          for i = 1, maxDets do
            for d = 1, stateDim do
              dets[i][t][d] = torch.rand(1):squeeze() * dRange[3][d] + dRange[1][d]
            end
          end
        end
      end

      -- detex 
      for t = 1, tmpwin do
        for j = 1, maxDets do
          if ada[1][t][j] == 0 then
            detex[1][t][j] = 0
          end
        end
      end
      -- print('m='..m)
      -- print('atrack') print(atrack)
      -- print('ada') print(ada)
      -- print('dets') print(dets)
      -- print('ex') print(ex)
      -- print('detex') print(detex)
      -- abort()
      -- print(alltr) 

      alltr = alltr:cat(atrack, 1)
      alldet = alldet:cat(dets, 1)
      allda = allda:cat(ada, 1)
      allex = allex:cat(ex, 1)
      alldetex = alldetex:cat(detex, 1)
      table.insert(allseqnames, seqName)
    end
    alltr = alltr:sub(2,-1)        -- batch*maxTargets x tempwin x D
    alldet = alldet:sub(2,-1)      -- batch*maxDets x tmpwin x D
    allda = allda:sub(2,-1)        -- batch*maxTargets x tempwin x nClasses
    allex = allex:sub(2,-1)        -- batch*maxTargets x tempwin
    alldetex = alldetex:sub(2,-1)  --batch x tempwin x maxDets
    -- print('alldet') print(alldet)
    -- print('allda') print(allda)
    -- print('allex') print(allex)
    -- print('alldexex') print(alldetex)
    -- abort()
    table.insert(trTracksTab, alltr)
    table.insert(trDetsTab, alldet)
    table.insert(trDasTab, allda)
    table.insert(trExTab, allex)
    table.insert(trDetExTab, alldetex)
    table.insert(trSeqNameTab, allseqnames)
  end
  -- print('trTracksTab') print(trTracksTab)
  -- print('trDetsTab') print(trDetsTab)
  -- print('trDasTab') print(trDasTab) 
  -- print('trExTab') print(trExTab[1])
  -- print('trDetExTab') print(trDetExTab)
  -- print('trSeqNameTab') print(trSeqNameTab)
  -- abort()
  for k,v in pairs(trTracksTab) do trTracksTab[k] = dataToGPU(trTracksTab[k]) end
  for k,v in pairs(trDetsTab) do trDetsTab[k] = dataToGPU(trDetsTab[k]) end
  for k,v in pairs(trDasTab) do trDasTab[k] = dataToGPU(trDasTab[k]) end
  for k,v in pairs(trExTab) do trExTab[k] = dataToGPU(trExTab[k]) end
  for k,v in pairs(trDetExTab) do trDetExTab[k] = dataToGPU(trDetExTab[k]) end
  -- for k,v in pairs(trSeqNameTab) do trSeqNameTab[k] = dataToGPU(trSeqNameTab[k]) end
  return trTracksTab, trDetsTab, trDasTab, trExTab, trDetExTab, trSeqNameTab
end
--------------------------------------------------------------
--------------
function preAllTestData(mode, sequences, norm)
  print('Preparing '..mode..' All data...')

  -- if norm == nil then norm = true end
  maxmin_State = {}  -- global var

  local TracksTab, DetsTab, DasTab ={}, {}, {}

  for _, seqName in pairs(sequences) do
    -- -- 1. get the raw data Tensor
    local tracks = getGtTracks(seqName)               -- nxFxD, n=1
    local maxtrackPerFrame, Fs = tracks:size(1), tracks:size(2)
    local detections = getDets(seqName, Fs)           -- mxFxD
    local maxDetPerFrame =detections:size(1)
    local das = getDAs(seqName, maxtrackPerFrame, Fs) -- nxFx(m+1), n=1
    -- print(das) abort()
    
    -- 2. get the max state value of tracks and detections
    GetMaxMinState(tracks, detections, seqName)

    -- 3. normalization min-max scaling
    local Ngt, Fgt, Dgt = (tracks)
    local maxX, minX = maxmin_State[seqName]['MaxX'], maxmin_State[seqName]['MinX']
    local maxY, minY = maxmin_State[seqName]['MaxY'], maxmin_State[seqName]['MinY']

    -- print('tracks') print(tracks)
    -- print('detections') print(detections) abort()
    if norm then
      if sopt.normalize_data == 1 then --max-min
        print('======== preAllTestData norm 1')
        -- print(maxmin_State)
        tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - minX)/(maxX - minX)
        detections[{{},{},{1}}] = (detections[{{},{},{1}}] - minX)/(maxX - minX)
        tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - minY)/(maxY - minY)
        detections[{{},{},{2}}] = (detections[{{},{},{2}}] - minY)/(maxY - minY)
      elseif sopt.normalize_data == 2 then  -- std-dev
        print('======== preAllTestData norm 2')
        tracks[{{},{},{1}}] = (tracks[{{},{},{1}}] - maxX/2)/maxX
        detections[{{},{},{1}}] = (detections[{{},{},{1}}] - maxX/2)/maxX
        tracks[{{},{},{2}}] = (tracks[{{},{},{2}}] - maxY/2)/maxY
        detections[{{},{},{2}}] = (detections[{{},{},{2}}] - maxY/2)/maxY
      end
    end

    -- print('tracks') print(tracks)
    -- print('detections') print(detections) abort()
    table.insert(TracksTab,tracks)    --(seqNum, N, F, D)
    table.insert(DetsTab,detections)  --(seqNum, M, F, D)
    table.insert(DasTab,das)          --(seqNum, N, F, M+1)
  end
  -- local nProc = 1
  -- if miniBatchSize>1 then nProc = opt.data_train else nProc = opt.data_valid end
  -- print('nProc='..nProc)
  -- local TRACKsTab, DETsTab, DAsTab, EXsTab, DetEXsTab, SeqNamesTab = processAllData(sequences, TracksTab ,DetsTab, DasTab, nProc, miniBatchSize)
  return TracksTab, DetsTab, DasTab
end

--------------------------------------------------------------
--------------
function preEXData(mode, sequences)
  print('Preparing '..mode..' EX data...')

  maxmin_State = {}  -- global var

  local TracksTab, DetsTab, DasTab ={}, {}, {}

  for _, seqName in pairs(sequences) do
    -- -- 1. get the raw data Tensor
    local tracks = getGtTracks(seqName)               -- nxFxD, n=1
    local maxtrackPerFrame, Fs = tracks:size(1), tracks:size(2)
    local detections = getDets(seqName, Fs)           -- mxFxD
    local maxDetPerFrame =detections:size(1)
    local das = getDAs(seqName, maxtrackPerFrame, Fs) -- nxFx(m+1), n=1

    GetMaxMinState(tracks, detections, seqName)

    -- print('tracks') print(tracks)
    -- print('detections') print(detections) abort()
    table.insert(TracksTab,tracks)    --(seqNum, N, F, D)
    table.insert(DetsTab,detections)  --(seqNum, M, F, D)
    table.insert(DasTab,das)          --(seqNum, N, F, M+1)

  end
  if miniBatchSize>1 then nProc = opt.data_train else nProc = opt.data_valid end
  print('nProc='..nProc)
  local DAsTab, EXsTab, SeqNamesTab = processEXData(sequences, TracksTab ,DetsTab,DasTab, nProc, miniBatchSize)
  return DAsTab, EXsTab, SeqNamesTab
end

--------------------------------------------------------------------------
--- preprocess one(target)-to-multi(detections) data
function processEXData(sequences, tracksTab ,detsTab, dasTab, nProc, mbSize)
  local trDasTab, trExTab, trSeqNameTab= {}, {}, {}, {}, {}, {}
  local seqNum = #sequences
  local tmpwin = opt.temp_win
  local birth_death = true
  for n=1,nProc do
    local allda = torch.zeros(1, tmpwin, nClasses)
    local allex = torch.zeros(1, tmpwin)
    local allseqnames = {}
    for m=1,mbSize do
      local randSeq= math.random(seqNum) -- select a sequence randomly
      local seqName = sequences[randSeq]
      local tracks = tracksTab[randSeq]                -- N x F x D
      local dets = detsTab[randSeq]                    -- M x F x D
      local das = dasTab[randSeq]                        -- N x F x M+1
  
      local N= tracks:size(1)
      local randTar = math.random(N)     -- select a target randomly form a sequence
    
      local cur_Tar_len = das[{{randTar},{},{nClasses}}]:eq(0):squeeze()
      local F = cur_Tar_len:sum()
      -- print(seqName, randTar,F) 

      local trajS = math.random(F-tmpwin)
      local trajE = trajS + tmpwin
      -- print('trajS..'..trajS..',trajE..'..trajE)

      -- deep copy!!!!!!!!!!!!
      local atrack = tracks[{{randTar},{trajS,trajE}}]:clone() -- 1 x F(tmpwin+1) x D
      -- print('atrack') print(atrack) 
      local dets = dets[{{},{trajS,trajE}}]:clone()            -- M x F(tmpwin+1) x D
      -- print('dets ') print(dets) 
      

      -- initialize ex
      local ex = torch.ones(1, tmpwin)

      -- birth/death : fix ex
      local d_rate = torch.rand(1)
      -- print(d_rate)
      local exS = math.random(tmpwin/4)
      local exE = tmpwin - torch.random(tmpwin-3)+1
      if d_rate[1]<opt.death_rate and TRAINING then
        if exS>1 then ex[{{1},{1,exS-1}}]=0 end
        if exE-exS>2 and exE<tmpwin then
          ex[{{1},{exE+1,tmpwin}}]=0
        end
      end
      -- fix det based on ex
      for t =1, tmpwin do
        if ex[1][t]==0 then
          for i = 1, opt.max_m do
            for d = 1, stateDim do
              if d%2==0 then
                dets[i][t+1][d] = torch.rand(1):squeeze() * maxmin_State[seqName]['MaxY']
              else
                dets[i][t+1][d] = torch.rand(1):squeeze() * maxmin_State[seqName]['MaxX']
              end
            end
          end
        end
      end

      -- calculate data assosication prob
      local newda = torch.zeros(1,tmpwin,nClasses)
      for t=1, tmpwin do
        local tar = atrack[1][t]:clone():reshape(1,stateDim)
        tar = dataToGPU(tar)
        local meas = dets[{{},{t+1}}]:clone():reshape(opt.max_m, stateDim)
        meas = dataToGPU(meas)
        local aprob = JPDA_nonorm(tar,meas)
        newda[{{1},{t}}]=aprob
        -- print(aprob)
      end
      

      -- print('m='..m)
      -- print(newda)
      -- print('ex') print(ex)
      -- abort()

      allda = allda:cat(newda, 1)
      allex = allex:cat(ex, 1)
      table.insert(allseqnames, seqName)
    end
    allda = allda:sub(2,-1)        -- batch*maxTargets x tempwin x nClasses
    allex = allex:sub(2,-1)        -- batch*maxTargets x tempwin
    -- print('allda') print(allda)
    -- print('allex') print(allex)
    -- abort()
    table.insert(trDasTab, allda)
    table.insert(trExTab, allex)
    table.insert(trSeqNameTab, allseqnames)
  end
  -- print('trDasTab') print(trDasTab)   -- [data_train] x batch*1 x tempwin x nClasses
  -- print('trExTab') print(trExTab)     -- [data_train] x batch*1 x tempwin 
  -- print('trSeqNameTab') print(trSeqNameTab)
  -- abort()
  for k,v in pairs(trDasTab) do trDasTab[k] = dataToGPU(trDasTab[k]) end
  for k,v in pairs(trExTab) do trExTab[k] = dataToGPU(trExTab[k]) end
  -- for k,v in pairs(trSeqNameTab) do trSeqNameTab[k] = dataToGPU(trSeqNameTab[k]) end
  return trDasTab, trExTab, trSeqNameTab
end