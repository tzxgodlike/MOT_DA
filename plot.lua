require 'gnuplot'

--------------------------------------------------------------------------
--- Plot pretty graphs
-- @param data 		A table containing the data series and their properties.
-- @param winID		Windows ID (Optional).
-- @param winTitle	Widnow title (Optional, also used for png file name)
-- @param rawStr 	gnu setup to be used (Optional)
-- @param save		Flag to save as PNG in addition
function plot(data, winID, winTitle, rawStr, save)
  if not isValidPlotData(data) then 
      print("Invalid plot data. Unable to plot.")
      return false 
  end	-- to avoid error with empty or invalid data

  if save == nil then save = false 
  else
    if (save==true or save>0) then save=true else save=false end
  end
  
  winID = winID or 1
  winTitle = winTitle or "unnamed"
  rawStr = rawStr or {}
  if type(rawStr) ~= 'table' then return end  --  GNU Plot commands must be passed in a table

  local gpterm = 'wxt'
  local enhancedString = 'enhanced dashed "arial,16"'
  gnuplot.raw(string.format('set term %s %s %d',gpterm, enhancedString, winID))
  gnuplot.raw(string.format('set term %s title "%s"',gpterm,winTitle))

  local yr = 8   -- yrange and yrange shift

  if opt.norm_std ~= nil and opt.norm_std > 0 then yr = 800 end
  if opt.norm_std ~= nil and opt.norm_std < 0 then yr = 1 end
  local ys = yr/2

  if opt.norm_mean ~= nil and opt.norm_mean == 0 then ys = 0 end
  local rangeStr = string.format("set xrange [%d:%d]; set yrange [%f:%f]",0,opt.temp_win+1,-ys,yr-ys)

  gnuplot.raw('set key outside left yrange [0:0.2]')
  gnuplot.raw(rangeStr)

  --     gnuplot.raw('set xlabel "frame"')
  --     print(data)

  --append additional options for formatting, etc...
  for _,v in pairs(rawStr) do gnuplot.raw(v) end

    -- if x available, display
  if xAvailable() then
    gnuplot.plot(data)
    gnuplot.plotflush()
  end
  

  -- divert to png if no X xAvailable  UPDATE: Always save as png if flag set
  if save then
    gnuplot.raw('set term pngcairo enhanced font "arial,10" fontscale 1.0 size 1000,400;')

    -- make sure output directory exists (TODO take care of global vars here)

    -- local _,_,modelName,modelSign = getCheckptFilename(modelName, opt, modelParams)
    -- local outDir = string.format('tmp/%s_%s',modelName, modelSign)
    -- if not lfs.attributes(outDir,'mode') then lfs.mkdir(outDir) end
    gnuplot.raw(string.format("set output \'../tmp/trainDA_r500_l2_n2_m4_d1/%s.png\'",winTitle))
    gnuplot.raw('set key outside left yrange [0:0.2]')
    gnuplot.raw(rangeStr)
    gnuplot.plot(data)
    gnuplot.raw("set output")
    -- gnuplot.raw("set term wxt")
    gnuplot.plotflush()
  end
end

--------------------------------------------------------------------------
--- Check data table for validity
-- @param data 		A table containing the data series and their properties.
function isValidPlotData(data)
  if type(data) ~= 'table' then return false end
  if #data==0 or data == nil then return false end

  local valid = true
  for k,l in pairs(data) do
    for m,n in pairs(l) do
      if torch.isTensor(n) then
        if n:nDimension() < 1 then valid = false end
        if n:nElement() < 1 then valid = false end
      end
    end
  end

  return valid
end

--------------------------------------------------------------------------
--- Converts a full data tensor to a 2D NxF for plotting.
-- @param data	The data. An NxFxD tensor
-- @param dim	State dimension to keep (default 1)
-- @return 	An NxF matrix
function getPlotTensor(data, dim)
  if data:nDimension() <= 2 then return data end 	-- nothing to do

  dim=dim or 1			-- by default, get first state dimension
  if opt~= nil and opt.plot_dim~= nil then dim=opt.plot_dim end
  if sopt~= nil and sopt.plot_dim~= nil then dim=sopt.plot_dim end
  --   dim = 3
  local N,F,D = getDataSize(data)
  --   data = data:narrow(3,dim,1):squeeze():reshape(N,F)  -- DO WE NEED SQUEEZE??
  data = data:narrow(3,dim,1):reshape(N,F)
  return data
end

--------------------------------------------------------------------------
--- Adds track entries to the plot table.
-- @param tracks	A NxFxD DoubleTensor containing the states
-- @param plotTab	An existing plot table to be added to (default nil).
-- @param ptype		The type of track. 1=GT, 2=Prediction
-- @param xvalues	[Optional] the x-values for plotting. Default 1,..,n
-- @param ex		A
-- @see getDetectionsPlotTab
function getTrackPlotTab(tracks, plotTab, ptype, xvalues, ex, tshift, DA, predStep)
  -- prepare for plotting
  -- tracks的维度变化： 2 x 4 x 1 -> 2 x 4
  tracks = getPlotTensor(tracks)	-- yields NxF matrix
  local N,F,D = getDataSize(tracks)
  local plotTab = plotTab or {}
  local ptype = ptype or 1	-- plot type 1 = GT, 2 = result
  tshift = tshift or 0

  local typestr = "GT"
  if ptype == 2 then typestr = "Upd" end
  if ptype == 3 then typestr = "Pred" end

  ----- EXPERIMENTAL --------
  local exTensor = {}
  if ex ~= nil then
    local nEx = 0
    if type(ex)=='table' then

      nEx = tabLen(ex)
      exTensor = torch.ones(N,nEx) -- NxF tensor with 1,2 existence classes
      for t,_ in pairs(ex) do
        mv,mi = torch.max(ex[t],2)
        exTensor[{{},{t}}] = mi
      end
    else -- ex is an NxFx2 tensor
      --     print(ex)
      --     abort()
      exTensor = torch.ones(N,F) -- NxF tensor with 1,2 existence classes
      for i=1,N do
      -- 	mv,mi = torch.max(ex[i],2)
      -- 	exTensor[{{i},{}}] = mi
      -- 	if ex[i]<.5 then exTensor
      end
      exTensor = ex:gt(0.5):reshape(N,F) -- only show existing
      --       exTensor = ex:gt(0.0):reshape(N,F) -- show all frames
      --       print(exTensor)
      --       abort()
    end

  end

  ----- EXPERIMENTAL --------
  local DATensor = {}
  local otracks = tracks:clone()
  if DA ~= nil then
    --     DATensor = torch.ones(N,F) --
    DATensor = DA:clone()
    --     for i=1,N do
    --       mv,mi = torch.max(DA[i],2)
    --       DATensor[{{i},{}}] = mi
    --     end
    otracks = orderTracks(tracks, DATensor)
    --     print(DATensor)
    --     abort()
  end
  ---------------------------
  local upToN = N
  --   local upToN = opt.max_n
  for id=1,upToN do
    local thisTrack = tracks:sub(id,id) -- yields 1xF tensor
    if torch.sum(torch.abs(thisTrack))~=0 then 	-- only if non-zero track
      -- find exframes      TODO: Needs attention!
      local exFrames = torch.linspace(1,F,F)[thisTrack:ne(0)]
      local trLength = exFrames[-1] - exFrames[1] + 1
      local plotFrames = torch.linspace(exFrames[1],exFrames[-1],trLength)+tshift
      -- print('exFrames') print(exFrames)
      -- print('trLength') print(trLength)
      -- print('plotFrames') print(plotFrames)
      -- print('(tracks[{{id},{exFrames[1],exFrames[-1]}}]):t()')print((tracks[{{id},{exFrames[1],exFrames[-1]}}]):t())
      -- abort()

      local linetype = ptype
      if ptype == 1 then linetype = 4 end -- GT dashed
      if ptype == 2 then linetype = 1 end -- state update solid
      if ptype == 3 then linetype = 2 end -- prediction dashed
      --       if id>opt.max_n then linetype = 0 end
      local lw = 1
      if ptype == 2 then lw = 2 end

      trname = string.format("%s %d",typestr, id)
      ls = string.format("with lines lw %d linetype %d linecolor rgbcolor %s",lw,linetype,getColorFromID(id))
      -- print('ls') print(ls)
      if xvalues then
        table.insert(plotTab, {trname,xvalues,(tracks[{{id},{}}]):t(),ls}) -- t() for vertical vec.
      else
        if ex == nil and DA == nil then
          -- 	table.insert(plotTab, {trname,(tracks[{{id},{}}]):t(),ls}) -- t() for vertical vec.
          table.insert(plotTab, {trname,plotFrames, (tracks[{{id},{exFrames[1],exFrames[-1]}}]):t(),ls}) -- t() for vertical vec.
        elseif ex~= nil and DA == nil then
          local thisTrackEx=exTensor[{{id},{}}]
          local exLen = thisTrackEx:size(2)
          -- 	  thisTrackEx[1][3]=1
          local insertX = torch.linspace(1,exLen,exLen)[thisTrackEx:eq(1)]+tshift
          local insertY = tracks[{{id},{1,-1}}][thisTrackEx:eq(1)]
          -- 	  print(insertY)


          if insertY:nElement()>1 then

            table.insert(plotTab, {trname, insertX, insertY, ls}) -- track thick lines
            ls = string.format("with points pt 1 linecolor rgbcolor %s",getColorFromID(id))
            -- 	    table.insert(plotTab, {trname, insertX, insertY, ls}) -- pluses for track state
          end
        elseif ex == nil and DA ~= nil then
          -- 	  local insertX = torch.linspace(1,F,F)+tshift
          -- 	  local insertY = tracks[{{id},{1,-1}}]:reshape(F)

          local loctrackY=torch.zeros(F)
          local loctrackX=torch.zeros(F)
          for t=1,F do
            local sid = DATensor[id][t]


            trname = string.format("%s %d",typestr, sid)
            insertX = torch.Tensor({t+tshift})
            insertY = torch.Tensor({tracks[id][t]})

            loctrackX[t] = insertX
            loctrackY[t] = torch.Tensor({tracks[sid][t]})

            -- 	    table.insert(plotTab, {trname, t+tshift, insertY, ls})
            ls = string.format("with points pt 2 ps 1 linecolor rgbcolor %s",getColorFromID(sid))
            table.insert(plotTab, {insertX, insertY, ls})

          end
          local loctrackX = torch.linspace(1,F,F)+tshift
          local loctrackY = otracks[{{id},{1,-1}}]:reshape(F)

          trname = string.format("track %d", id)
          ls = string.format("with lines lw %d linetype %d linecolor rgbcolor %s",lw,linetype,getColorFromID(id))
          table.insert(plotTab, {trname, loctrackX, loctrackY, ls})

        elseif ex ~= nil and DA ~= nil then
          local thisTrackEx=exTensor[{{id},{}}]
          local exLen = thisTrackEx:size(2)
          -- 	  thisTrackEx[1][3]=1
          local insertX = torch.linspace(1,exLen,exLen)[thisTrackEx:eq(1)]+tshift
          local insertY = otracks[{{id},{1,-1}}][thisTrackEx:eq(1)]
          local f = exLen

          if insertY:nElement()>1 then

            table.insert(plotTab, {trname, insertX, insertY, ls})
            ls = string.format("with points pt 1 linecolor rgbcolor %s",getColorFromID(id))
            table.insert(plotTab, {trname, insertX, insertY, ls})
          end

        end

      end
    end
  end
  return plotTab
end

--------------------------------------------------------------------------
--- Get times and state series for detections (all in one vec)
function getDetectionsVec(detections, thr)
  thr = thr or 0
  detections = getPlotTensor(detections)
  local detMask = torch.ne(detections,thr) -- find exisiting detections
  local nDet = torch.sum(detMask) -- number of detections
  local detX = torch.zeros(nDet)
  local detT = torch.zeros(nDet)
  local cnt = 0
  for id=1,detections:size(1) do
    for t=1,detections:size(2) do
      if detections[id][t] ~= thr then
        cnt=cnt+1
        detX[cnt]=detections[id][t]
        detT[cnt]=t
      end

    end
  end
  return detT,detX
end
--------------------------------------------------------------------------
--- Put detections in plotting table.
-- @param detections	A NxF DoubleTensor containing the x coordinates.
-- @param plotTab	An existing plot table to be added to (default nil).
-- @param thr		Threshold to discard detections.
-- @param da 		(Optional). Color detections if provided.
function getDetectionsPlotTab(detections, plotTab, thr, da,tshift,vdet)
  plotTab = plotTab or {}
  thr = thr or 0
  tshift=tshift or 0

  detections = getPlotTensor(detections)

  -- if we have associations, then plot with according colors
  if da then
    local Ndet,Fdet = getDataSize(detections)
    local maxID = torch.max(da)
    for i=maxID,-1,-1 do
      local IDmask = torch.eq(da,i) 	-- binary mask for current ID
      local detMask = torch.ne(detections,0)	-- binary mask for existing dets
      -- print('detections')      print(detections)
      -- print('IDmask')      print(IDmask)
      -- print('detMask') print(detMask)
      IDmask = torch.cmul(IDmask:int(), detMask:int()):byte()
      -- print('IDmask')      print(IDmask)
      -- abort()
      local N = torch.sum(IDmask)	-- how many detections carry this ID
      if N > 0 and i ~= 0 then 		-- skip if no IDs found (N==0)

        local locs = torch.Tensor(N)
        local times = torch.Tensor(N)
        local cnt=0

        -- WARNING: This might be slow!
        for t=1,Fdet do
          for d=1,Ndet do
            if IDmask[d][t] > 0  then
              cnt=cnt+1
              locs[cnt] = detections[d][t]
              times[cnt] = t+tshift
            end
          end
        end
        legName =  string.format("Det %d",i)
        ls = string.format("with points pt 7 ps 1 linecolor rgbcolor %s",getColorFromID(i)) -- detection dots
        if vdet then
          legName =  string.format("VDet %d",i)
          ls = string.format("with points pt 6 ps .75 linecolor rgbcolor %s",getColorFromID(i)) -- virtual detection O's
        end
        -- 	if i==-1 or i == Ndet then
        if i<=0 then
          legName =  string.format("FA Det")
          ls = string.format("with points pt 6 ps 0.5 linecolor rgbcolor %s",getColorFromID(14))
        end
        table.insert(plotTab, {legName, times, locs, ls})
      end -- if N>0
    end -- for 1,maxID


  else	-- otherwise all in one vec
    local detT,detX = getDetectionsVec(detections, thr)
    ls = string.format("with points pt 7 ps 0.25 linecolor rgbcolor %s",getColorFromID(14))
    table.insert(plotTab, {"Det",detT+tshift,detX,ls})
  end
  -- print('plotTab') print(plotTab)
  -- abort()
  return plotTab
end

--------------------------------------------------------------------------
--- Put data association connections as vertical lines
-- TODO: doc
function getDAPlotTab(tracks, detections, plotTab, da, predEx, tshift, predDA)
  plotTab = plotTab or {}
  tshift=tshift or 0

  detections = getPlotTensor(detections)
  tracks = getPlotTensor(tracks)

  local dim=dim or 1			-- by default, get first state dimension
  if opt~= nil and opt.plot_dim~= nil then dim=opt.plot_dim end
  if sopt~= nil and sopt.plot_dim~= nil then dim=sopt.plot_dim end


  --   print(tracks)
  --   print(GTSTATE)
  --   tracks = getPlotTensor(GTSTATE)

  local predExBin = nil

  local N,F,D = getDataSize(tracks)

  local upToN = N
  local upToN = opt.max_n

  --   print(da)
  --   print(labels)
  if predEx ~= nil then predExBin = getPredEx(predEx) else predExBin=torch.ones(N,F) end
  for t=1,F do
    for i=1,upToN do
      local detIDToUpdate = da[i][t]
      --       print(detIDToUpdate)
      if predExBin[i][t] > 0 and detIDToUpdate <= detections:size(1) then
        --       if detIDToUpdate <= detections:size(1) then
        local x = tracks[i][t]
        -- 	x = GTSTATE[i][t+1][1]

        local d = detections[detIDToUpdate][t]

        local randShift=(math.random()-.5)*.1
        local randShift=0
        local trX = t+1+tshift
        local detX = t+1+randShift+tshift
        local connX = torch.Tensor({trX,detX})

        local connY = torch.Tensor({x,d})


        local psize=1
        if d~=0 then
          local DAprob = torch.exp(predDA[i][t][detIDToUpdate])
          -- 	  DAprob = 1
          -- 	  ls = string.format("with lines lw %d linetype %d linecolor rgbcolor %s",1,1,getColorFromID(i))
          gnuplot.raw("set palette defined (0 'white', 1 'black')")
          ls = string.format("with lines lw %f linetype %d linecolor palette frac %f",3*DAprob,1,DAprob)
          -- 	  gnuplot.raw("set palette rgbformulae 21,22,23")
          -- 	  gnuplot.raw("unset colorbox")

          table.insert(plotTab, {connX, connY,ls}) -- connections
        end

        local pttype = 2
        local assY = torch.Tensor({d})
        local assX = torch.Tensor({detX})
        if d == 0 then pttype = 6; psize=2; assY = torch.Tensor({x}) end
        ls = string.format("with points pt %d ps %d linetype %d linecolor rgbcolor %s",pttype,psize,1,getColorFromID(i))
        table.insert(plotTab, {assX, assY,ls})     -- X's
      elseif detIDToUpdate > detections:size(1) then
        if exlabels~= nil and exlabels[i][t+1]~=0 then
          local pttype=6
          local psize=2
          local assX = torch.Tensor({t+1})
          local GTTracks

          local assY = torch.Tensor({GTSTATE[i][t+1][dim]})
          -- 	print(assX, assY)
          -- 	sleep(1)
          ls = string.format("with points pt %d ps %d linetype %d linecolor rgbcolor %s",pttype,psize,1,getColorFromID(i))
          table.insert(plotTab, {assX, assY,ls})     -- O's
        end
      end
    end
  end


  -- correct DA
  local psize = 2
  local pttype = 4
  for t=1,F do
    for i=1,upToN do

      local dX = torch.Tensor({t})
      local dY = torch.Tensor({0})
      if labels[i][t] <= detections:size(1) then
        dY = torch.Tensor({detections[labels[i][t]][t]})

        ls = string.format("with points pt %d ps %d linetype %d linecolor rgbcolor %s",pttype,psize,1,getColorFromID(i))
        --       print(dX, dY, ls)
        table.insert(plotTab, {dX, dY, ls})
      end
    end
  end
  --   print(labels)
  --   abort()

  return plotTab
end

--- Plot existence probabilities
function getExPlotTab(plotTab,predEx, tshift)
  tshift=tshift or 0

  local N,F = getDataSize(predEx)
  local exFrames = torch.linspace(1,F,F)+tshift
  for id=1,N do
    local probs = predEx[id]:squeeze()-0.5
    trname = string.format("Exist %d", id)
    ls = string.format("with lines lw %d linetype %d linecolor rgbcolor %s",1,3,getColorFromID(id))
    table.insert(plotTab, {trname, exFrames, probs, ls})

  end
  return plotTab
end

--------------------------------------------------------------------------
--- Get a smoothed version of the training loss for plotting
-- @param i		iteration
-- @param plotLossFreq	plotting frequency
-- @param trainLosses	A table containing (iteration number, training loss) pairs
function getLossPlot(i, plotLossFreq, trainLosses)

  local addOne = 0-- plus one for iteration=1
  local plot_loss = torch.Tensor(math.floor(i / (plotLossFreq))+addOne)
  local plot_loss_x = torch.Tensor(math.floor(i / (plotLossFreq))+addOne)
  if i==1 then plot_loss, plot_loss_x = torch.ones(1), torch.ones(1) end

  local cnt = 0
  local tmp_loss = 0
  local globCnt = 0
  for a,b in pairs(trainLosses) do
    cnt = cnt+1
    tmp_loss = tmp_loss + b
    -- compute avaerage and reset
    --     if a==1 or (a % plotLossFreq) == 0 then
    if (a % plotLossFreq) == 0 then
      globCnt = globCnt + 1
      tmp_loss = tmp_loss / cnt
      plot_loss[globCnt] = tmp_loss
      plot_loss_x[globCnt] = a
      cnt = 0
      tmp_loss = 0
    end
  end
  --   plot_loss_x, plot_loss = getValLossPlot(trainLosses)

  return plot_loss_x, plot_loss

end



--------------------------------------------------------------------------
--- Get a smoothed version of the training loss for plotting获得训练损失的平滑版本
function getValLossPlot(val_losses)
  local tL = tabLen(val_losses)
  --   local plot_val_loss = torch.Tensor(tL)
  --   local plot_val_loss_x = torch.Tensor(tL)

  local plot_val_loss = {}
  local plot_val_loss_x = {}

  local orderedKeys = {}
  for k in pairs(val_losses) do table.insert(orderedKeys, k) end
  table.sort(orderedKeys)

  --   print(val_losses)

  local cnt2 = 0
  for cnt=1,tL do
    --     if val_losses[orderedKeys[cnt]] > 0 then
    cnt2=cnt2+1
    plot_val_loss[cnt2]=val_losses[orderedKeys[cnt]]
    plot_val_loss_x[cnt2]=orderedKeys[cnt]
    --     end
  end

  --   print(plot_val_loss)

  --   if cnt2==0 then
  --     plot_val_loss=torch.Tensor(0)
  --     plot_val_loss_x=torch.Tensor(0)
  --   else
  plot_val_loss=torch.Tensor(plot_val_loss)
  plot_val_loss_x=torch.Tensor(plot_val_loss_x)
  --   end
  return plot_val_loss_x, plot_val_loss
end



--------------------------------------------------------------------------
--- Prints depending on the debug level set in opt.verbose
-- @param message 	The message to be printed
-- @param vl		The verbosity level (0=none, 1=warning, 2=info, 3=debug)
function pm(message, vl)
  vl=vl or 2
  if vl <= opt.verbose then
    print(message)
  end
end

--------------------------------------------------------------------------
--- Prints all options (TODO Needs prettyfying)
function printOptions(opt)
  local modelOpt = {
    'rnn_size','num_layers'
  }
  local hide = {
    ['gpuid']=0,['verbose']=0,['savefile']=0,['checkpoint_dir']=0,
    ['print_every']=0,['plot_every']=0,['eval_val_every']=0,['profiler']=0
  }
  hide = {}

  for k,v in pairs(opt) do
    if hide[k] == nil then
      pm(string.format("%22s  %s",k,tostring(v)),2) -- to string to handle both num. and str. types
    end
  end
end

--------------------------------------------------------------------------
--- Get color for plotting.
-- @param id	Target ID.
function getColorFromID(id)

  local colTable = {
    'brown_025','blue_000','green_075','brown_050',
    'red_100','green_000','blue_050','brown_075',
    'red_025','blue_025','brown_000','red_075',
    'blue_075','green_050','blue_100','red_000',
    'green_100','red_050','brown_100','green_025'
  }

  --   return col
  modid = id % table.getn(colTable) + 1
  return colTable[modid]

end

function getGlobalLoss(maxepoch, train_losses, val_losses, eval_val_every)

  local plot_train_loss={}
  local plot_val_loss={}
  local tL_train= tabLen(train_losses)
  local val_x=maxepoch-maxepoch%eval_val_every
  local plot_train_loss_x = torch.linspace(1,tL_train,tL_train)
  local plot_val_loss_x = plot_train_loss_x
  local plot_val_loss_x = torch.linspace(eval_val_every, val_x, val_x/eval_val_every)
  local cnt=0
  for k=1,tL_train do
    if (k%eval_val_every)==0 then
      cnt =cnt+1
      plot_val_loss[cnt]=val_losses[k]
    end
  end
  -- print('val_losses') print(val_losses)
  plot_train_loss=torch.Tensor(train_losses)
  plot_val_loss=torch.Tensor(plot_val_loss)
  -- print('plot_val_loss') print(plot_val_loss) abort()
  local maxloss=torch.max(plot_train_loss)
  if maxloss>torch.max(plot_val_loss) then
    maxloss=torch.max(plot_val_loss)
  end

  local minloss=torch.min(plot_train_loss)
  if minloss>torch.min(plot_val_loss) then
    minloss=torch.min(plot_val_loss)
  end

  local lossPlotTab = {}
  table.insert(lossPlotTab, {"Trng loss",plot_train_loss_x,plot_train_loss, 'with linespoints lt 1'})
  table.insert(lossPlotTab, {"Vald loss",plot_val_loss_x, plot_val_loss, 'with linespoints lt 3'})

  return lossPlotTab,maxloss,minloss
end

function plotLoss(me, eval_every)
  LossPlotTab, maxloss, minloss = getGlobalLoss(opt.max_epochs, train_losses, val_losses, opt.eval_val_every)
  -- print('LossPlotTab') print(LossPlotTab)
  local winTitle = string.format('Loss-%d-%d',1,opt.max_epochs)
  gnuplot.raw('set term pngcairo enhanced font "arial,10" fontscale 1.0 size 1000,400;')

-------LOSS make sure output directory exists (TODO take care of global vars here)
  local _,_,modelName,modelSign = getCheckptFilename(modelName, opt, modelParams)
  local rootDir = getRNNTrackerRoot()
  local outDir = string.format('%s/out/%s_%s',rootDir, modelName, modelSign)
  if not lfs.attributes(outDir) then lfs.mkdir(outDir) end
  gnuplot.raw(string.format("set output \"%s/%s.png\"",outDir, winTitle))
  gnuplot.raw(string.format('set yrange [%f:%f]',0, maxloss+0.01))
  gnuplot.plot(LossPlotTab)
  gnuplot.raw("set output")
-----LOSS gnuplot.raw("set term wxt")
  gnuplot.plotflush()
end
