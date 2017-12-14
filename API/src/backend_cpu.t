local b = {}
local S = require("std")
local c = require('config')
local la = require('linalg_cpu')
local s = require("simplestring")

local C = terralib.includecstring [[
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]
local I = require('ittnotify')
-- error()

b.name = 'CPU'
b.numthreads = 1 -- DO NOT CHANGE THIS

b.threadarg = {}
b.threadarg_val = 1

-- TODO atm, this is defined here AND in util.t because util.t depends on the
-- backend file. Need to refactor to fix this somehow
-------------------------------------------------------------------------------
-- MACRO FOR TIMED EXECUTION START
-- displays execution time in milliseconds
-- Example usage:
-- util.texec("step(): PCGStep1", true,
--   gpu.PCGStep1(pd)
-- )
local texec = function(msg, printOutput, stmt)
-- return quote
-- var a = 1
-- end
if printOutput then
  return quote
    var start : C.timespec
    var stop : C.timespec

    C.clock_gettime(C.CLOCK_MONOTONIC, &start)
    [stmt]
    C.clock_gettime(C.CLOCK_MONOTONIC, &stop)

    var elapsed : double
    elapsed = 1000*(stop.tv_sec - start.tv_sec)
    elapsed = elapsed + (stop.tv_nsec - start.tv_nsec)/[double](1e6)

    C.printf("TEXEC: %s t = %f ms\n", [msg], elapsed)
  end
else
  return quote [stmt] end
end

end
-- MACRO FOR TIMED EXECUTION END
-------------------------------------------------------------------------------


--------------------------- Timing stuff start
-- TODO Array should be a separate helper class
-- TODO only used in some timer-related stuff below, so make local there and put in appropriate file
function Array(T,debug)
    local struct Array(S.Object) {
        _data : &T;
        _size : int32;
        _capacity : int32;
    }
    function Array.metamethods.__typename() return ("Array(%s)"):format(tostring(T)) end
    local assert = debug and S.assert or macro(function() return quote end end)
    terra Array:init() : &Array
        self._data,self._size,self._capacity = nil,0,0
        return self
    end
    terra Array:reserve(cap : int32)
        if cap > 0 and cap > self._capacity then
            var oc = self._capacity
            if self._capacity == 0 then
                self._capacity = 16
            end
            while self._capacity < cap do
                self._capacity = self._capacity * 2
            end
            self._data = [&T](S.realloc(self._data,sizeof(T)*self._capacity))
        end
    end
    terra Array:initwithcapacity(cap : int32) : &Array
        self:init()
        self:reserve(cap)
        return self
    end
    terra Array:__destruct()
        assert(self._capacity >= self._size)
        for i = 0ULL,self._size do
            S.rundestructor(self._data[i])
        end
        if self._data ~= nil then
            C.free(self._data)
            self._data = nil
        end
    end
    terra Array:size() return self._size end
    
    terra Array:get(i : int32)
        assert(i < self._size) 
        return &self._data[i]
    end
    Array.metamethods.__apply = macro(function(self,idx)
        return `@self:get(idx)
    end)
    
    terra Array:insertNatlocation(idx : int32, N : int32, v : T) : {}
        assert(idx <= self._size)
        self._size = self._size + N
        self:reserve(self._size)

        if self._size > N then
            var i = self._size
            while i > idx do
                self._data[i - 1] = self._data[i - 1 - N]
                i = i - 1
            end
        end

        for i = 0ULL,N do
            self._data[idx + i] = v
        end
    end
    terra Array:insertatlocation(idx : int32, v : T) : {}
        return self:insertNatlocation(idx,1,v)
    end
    terra Array:insert(v : T) : {}
        return self:insertNatlocation(self._size,1,v)
    end
    terra Array:remove(idx : int32) : T
        assert(idx < self._size)
        var v = self._data[idx]
        self._size = self._size - 1
        for i = idx,self._size do
            self._data[i] = self._data[i + 1]
        end
        return v
    end

    -- We need to check if T is a struct because if it is (e.g. Event from below)
    -- and no __eq method is defined for it, then we get a compiler error from
    -- the equality check.
    if not T:isstruct() or T == s.String then
        terra Array:indexof(v : T) : int32
            for i = 0LL,self._size do
                if (v == self._data[i]) then
                    return i
                end
            end
            return -1
        end
        terra Array:contains(v : T) : bool
            return self:indexof(v) >= 0
        end
    end
	
    return Array
end

local Array = S.memoize(Array)


-- TODO what is this? its only used in the next few lines? can we make this local to the Timer "class"?
-- TODO need to refactor parts of event stuff that are common with other backends
-- TODO refactor all the timer class to be independent of Opt, it could be a simple
-- and independent terra helper library instead!
local MAXNAMELENGTH = 100
local struct Event {
	starttime : C.timespec
	endtime : C.timespec
	duration : double -- unit: ms
        -- IMPOERTANT: The value of MAXNAMELENGTH had significant impact on overall
        -- program performance, if 'eventName' was declared as 'int[MAXNAMELENGTH]',
        -- with the current implementation (&int8) the performance issues are gone
        -- but we have to live live with a (small) memory leak for the moment.
	eventName : &int8
}
Event_MAXNAMELENGTH = MAXNAMELENGTH
b.Event = Event

terra Event:calcElapsedTime()
  var elapsed : double
  elapsed = 1000*(self.endtime.tv_sec - self.starttime.tv_sec)
  elapsed = elapsed + [double](self.endtime.tv_nsec - self.starttime.tv_nsec)/([double](1e6))
  self.duration = elapsed
end
terra Event:getName()
-- MUST return a &int8 pointer to the first char in the name, regardless of the
-- implementation.
  return &(self.eventName[0])
end


local struct Timer {
	eventList : &Array(Event)
}
b.Timer = Timer




terra Timer:init() 
	self.eventList = [Array(Event)].alloc():init()
end

terra Timer:cleanup()
	self.eventList:delete()
end 


terra Timer:startEvent(name : rawstring, eventptr : &Event)
    -- TODO fix memory leak
    -- TODO this is stupid! We shouldn't set an events name  when the timer
    -- started, but instead when the Event is declared/created.
    eventptr.eventName = [&int8](C.malloc( Event_MAXNAMELENGTH ))
    C.memcpy(eventptr:getName(), name, Event_MAXNAMELENGTH*sizeof(int8))

    C.clock_gettime(C.CLOCK_MONOTONIC, &((@eventptr).starttime))
end


terra Timer:endEvent(eventptr : &Event, dummy : int)
-- dummy arg is required in mt backend
    C.clock_gettime(C.CLOCK_MONOTONIC, &((@eventptr).endtime))

    self.eventList:insert(@eventptr)
end

-- TODO only used in next function, so make local there
terra isprefix(pre : rawstring, str : rawstring) : bool
    if @pre == 0 then return true end
    if @str ~= @pre then return false end
    return isprefix(pre+1,str+1)
end

for k,v in pairs(s) do print(k,v) end
terra Timer:evaluate()
	if ([c._opt_verbosity > 0]) then
          var aggregateTimingInfo = [Array(tuple(float,int))].salloc():init()
          var aggregateTimingNames = [Array(s.String)].salloc():init()

          for i = 0,self.eventList:size() do
            var event = self.eventList(i);
            var eventName : s.String = event:getName()

            event:calcElapsedTime()

            var index =  aggregateTimingNames:indexof(eventName)
            if index < 0 then
              aggregateTimingNames:insert(eventName)
              aggregateTimingInfo:insert({event.duration, 1})
            else
              aggregateTimingInfo(index)._0 = aggregateTimingInfo(index)._0 + event.duration
              aggregateTimingInfo(index)._1 = aggregateTimingInfo(index)._1 + 1
            end
          end


            C.printf("asdfasdfasdfasdfasdf %d\n", aggregateTimingNames:size())
          C.printf(		"--------------------------------------------------------\n")
          C.printf(		"        Kernel        |   Count  |   Total   | Average \n")
          C.printf(		"----------------------+----------+-----------+----------\n")
          for i = 0, aggregateTimingNames:size() do
              C.printf(	"----------------------+----------+-----------+----------\n")
              C.printf(" %-20s |   %4d   | %8.3fms| %7.4fms\n", [rawstring](aggregateTimingNames(i)), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1)
          end

          C.printf(		"--------------------------------------------------------\n")
          C.printf("TIMING ")
          for i = 0, aggregateTimingNames:size() do
              var n = aggregateTimingNames(i)
              if isprefix("PCGInit1",n) or isprefix("PCGStep1",n) or isprefix("overall",n) then
                  C.printf("%f ",aggregateTimingInfo(i)._0)
              end
          end

          C.printf("\n")
          -- TODO: Refactor timing code
          var linIters = 0
          var nonLinIters = 0
          for i = 0, aggregateTimingNames:size() do
              var n = aggregateTimingNames(i)
              if isprefix("PCGInit1",n) then
                  linIters = aggregateTimingInfo(i)._1
              end
              if isprefix("PCGStep1",n) then
                  nonLinIters = aggregateTimingInfo(i)._1
              end
          end
          var linAggregate : float = 0.0f
          var nonLinAggregate : float = 0.0f
          for i = 0, aggregateTimingNames:size() do
              var n = aggregateTimingInfo(i)._1
              if n == linIters then
                  linAggregate = linAggregate + aggregateTimingInfo(i)._0
              end
              if n == nonLinIters then
                  nonLinAggregate = nonLinAggregate + aggregateTimingInfo(i)._0
              end
          end
          C.printf("Per-iter times ms (nonlinear,linear): %7.4f\t%7.4f\n", linAggregate, nonLinAggregate)
          end
    
end
--------------------------- Timing stuff end

---------------- ReduceVar start
b.ReduceVar = &&opt_float
b.ReduceVarHost = b.ReduceVar

b.ReduceVar.allocate = function(variable)
  return quote
            [variable] = [&&opt_float](C.malloc(sizeof([&&opt_float]) * (b.numthreads+1)))
            for k = 0,b.numthreads+1 do
              [variable][k] = [&opt_float](C.malloc(sizeof(opt_float)))
            end
         end
end
b.ReduceVarHost.allocate2 = b.ReduceVar.allocate

b.ReduceVar.free = function(variable)
  return quote
            for k = 0,b.numthreads+1 do
              C.free([variable][k])
            end
            C.free([variable])
         end
end
b.ReduceVarHost.free = b.ReduceVar.free

b.ReduceVar.getDataPtr = function(varquote, k)
  return `[varquote][k]
end

b.ReduceVar.getData = function(varquote, k)
  return `@([varquote][k])
end
b.ReduceVarHost.getData2 = b.ReduceVar.getData

b.ReduceVar.setToConst = function(varquote, val)
  return quote
           for k = 0,b.numthreads+1 do
             C.memset([&opaque]([varquote][k]), val, sizeof(opt_float))
           end
         end
end

b.ReduceVar.memcpyDevice2Host = function(targetquote, sourcequote)
  print(targetquote)
    return quote
             for k = 0,b.numthreads+1 do
               C.memcpy([&opaque]([targetquote][k]), [&opaque]([sourcequote][k]), sizeof(opt_float))
             end
           end
  end
b.ReduceVar.memcpyDevice = b.ReduceVar.memcpyDevice2Host

b.ReduceVar.reduceAllThreads = function(varquote)
  return quote
            @([varquote][0]) = 0.0
            for k = 1,b.numthreads+1 do
              @([varquote][0]) = @([varquote][0]) + @([varquote][k])
            end
         end
end
b.ReduceVarHost.reduceAllThreads2 = b.ReduceVar.reduceAllThreads
---------------- ReduceVar end
-- TODO alternative version of reduceVar below yielded no performance improvment
-- ---------------- ReduceVar start
-- b.ReduceVar = &opt_float
-- b.ReduceVarHost = b.ReduceVar

-- b.ReduceVar.allocate = function(variable)
--   return quote
--               [variable] = [&opt_float](C.malloc(sizeof(opt_float)))
--          end
-- end
-- b.ReduceVarHost.allocate2 = b.ReduceVar.allocate

-- b.ReduceVar.free = function(variable)
--   return quote
--             C.free([variable])
--          end
-- end
-- b.ReduceVarHost.free = b.ReduceVar.free

-- b.ReduceVar.getDataPtr = function(varquote, k)
--   return `[varquote]
-- end

-- b.ReduceVar.getData = function(varquote, k)
--   return `@[varquote]
-- end
-- b.ReduceVarHost.getData2 = b.ReduceVar.getData

-- b.ReduceVar.setToConst = function(varquote, val)
--   return quote
--            @[varquote] = val
--          end
-- end

-- b.ReduceVar.memcpyDevice2Host = function(targetquote, sourcequote)
--   print(targetquote)
--     return quote
--                C.memcpy([&opaque]([targetquote]), [&opaque]([sourcequote]), sizeof(opt_float))
--            end
--   end
-- b.ReduceVar.memcpyDevice = b.ReduceVar.memcpyDevice2Host

-- b.ReduceVar.reduceAllThreads = function(varquote)
--   return quote
--          end
-- end
-- b.ReduceVarHost.reduceAllThreads2 = b.ReduceVar.reduceAllThreads
-- ---------------- ReduceVar end

-- atomicAdd START
-- if c.opt_float == float then
    -- local terra atomicAddSync(sum : &float, value : float, offset : int)
    local terra atomicAddSync(sum : &opt_float, value : opt_float, offset : int)
      @sum = @sum + value
    end
    -- local terra atomicAddNosync(sum : &float, value : float)
    local terra atomicAddNosync(sum : &opt_float, value : opt_float)
      @sum = @sum + value
    end
    b.atomicAdd_sync = atomicAddSync
    b.atomicAdd_nosync = atomicAddNosync
-- end
-- atomicAdd END

local terra warpReduce(val : opt_float) 

  return val;

end
b.warpReduce = warpReduce


-- landeid() is the 'local' id of a thread within a warp
terra b.laneid()
  return 0
end

-- allocate, memset and memcpy START
b.allocateHost = macro(function(ptrptr, numbytes, datatype)
  return quote @ptrptr = [&datatype:astype()](C.malloc(numbytes)) end -- astype is required for some reason, see new.t in terra tests
end)

b.allocateDevice = macro(function(ptrptr, numbytes, datatype)
  return quote @ptrptr = [&datatype:astype()](C.malloc(numbytes)) end
end)

b.memsetDevice = macro(function(ptr, value, numbytes)
  return `C.memset([&opaque](ptr), value, numbytes)
end)

b.memcpyDevice = macro(function(targetptr, sourceptr, numbytes)
  return `C.memcpy(targetptr, sourceptr, numbytes)
end)

b.memcpyDevice2Host = macro(function(targetptr, sourceptr, numbytes)
  return `C.memcpy(targetptr, sourceptr, numbytes)
end)

b.memcpyHost2Device = macro(function(targetptr, sourceptr, numbytes)
  return `C.memcpy(targetptr, sourceptr, numbytes)
end)
-- allocate, memset and memcpy END


-- TODO doesn't work with double precision
-- linalg stuff START
-- TODO make sure that these functions work with opt_float
local function insertMatrixlibEntries(PlanData_t)
-- just inserting garbage but these entries are needed for cuda backend
  PlanData_t.entries:insert {"handle", &opaque }
  PlanData_t.entries:insert {"desc", &opaque }
end
b.insertMatrixlibEntries = insertMatrixlibEntries


b.computeNnzPatternATA = la.computeNnzPatternATA
b.computeATA = la.computeATA


b.computeAT = la.computeAT
b.computeNnzPatternAT = la.computeNnzPatternAT


b.applyAtoVector = la.applyAtoVector

b.computeBoundsA = la.computeBoundsA




local terra initMatrixStuff(handlePtr : &opaque, descrPtr : &opaque)
-- this function needs to do some stuff in cuda backend, but not here.
end
b.initMatrixStuff = initMatrixStuff
-- linalg stuff END

function b.makeIndexInitializer(Index, dims, dimnames, fieldnames)

    local initIndex
    local dimnames = "xyz"
    local dimensionality = #dims

    local arglist = terralib.newlist()
    for k = 1, dimensionality do
      arglist:insert( symbol(int,fieldnames[k]) )
    end
    for k,v in pairs(arglist)  do print(k,v) end -- debug

    terra initIndex(self : &Index, [arglist]) : bool -- add 'x', 'y' and 'z' field to the index
        escape
            for k, arg in pairs(arglist) do
              emit quote
                self.[fieldnames[k]] = [arg]
              end
            end
        end  
        return true
    end
    print(dims) -- debug
    for k,v in pairs(dims)  do print(k,v) end -- debug
    print(initIndex)

    return initIndex

end

b.getKernelArglist = terralib.memoize(function(indexspace) -- TODO should this result be cached or do we need new symbols all the time?
    local numdims = indexspace:getDimensionality()
    print(numdims)
    local arglist = terralib.newlist()
    for k = 1, numdims do
        arglist:insert(symbol(int, 'x' .. tostring(k)))
    end
    -- local arglist = terralib.newlist()
    return arglist
end)


-- IMAGE SPECIALIZATION START
function b.make_Image_initGPU(imagetype_terra)
    local initGPU = terra(self : &imagetype_terra)
        var data : &uint8 -- we cast this to the correct type later inside setGPUptr

        b.cd( b.allocateDevice(&data, self:totalbytes(), uint8) )
        b.cd( b.memsetDevice(data, 0, self:totalbytes()) )

        self:initFromGPUptr(data) -- (short explanataion): set self.data = data (and cast to appropriate ptr-type)
    end
    initGPU:setname('Image.initGPU')

   return initGPU
end

function b.make_Image_metamethods__apply(imagetype_terra, indextype_terra, vectortype_terra, loadAsVector, VT)
-- TODO move this back to o.t
    local metamethods__apply
    if loadAsVector then
        metamethods__apply = terra(self : &imagetype_terra, idx : indextype_terra) : vectortype_terra
            var a = VT(self.data)[idx:tooffset()]
            return @[&vectortype_terra](&a)
        end
    else
        metamethods__apply = terra(self : &imagetype_terra, idx : indextype_terra) : vectortype_terra
            var r = self.data[idx:tooffset()]
            return r
        end
    end
    metamethods__apply:setname('Image.metamethods.__apply')

    return metamethods__apply
end

function b.make_Image_metamethods__update(imagetype_terra, indextype_terra, vectortype_terra, loadAsVector, VT)
-- TODO move this back to o.t
    local metamethods__update
    if loadAsVector then
        metamethods__update = terra(self : &imagetype_terra, idx : indextype_terra, v : vectortype_terra)
            VT(self.data)[idx:tooffset()] = @VT(&v)
        end
    else
        metamethods__update = terra(self : &imagetype_terra, idx : indextype_terra, v : vectortype_terra)
            self.data[idx:tooffset()] = v
        end
    end
    metamethods__update:setname('Image.metamethods.__update')

    return metamethods__update
end

function b.make_Image_atomicAddChannel(imagetype_terra, indextype_terra, scalartype_terra)
  local atomicAddChannel = terra(self : &imagetype_terra, idx : indextype_terra, c : int32, value : scalartype_terra)
      var addr : &scalartype_terra = &self.data[idx:tooffset()].data[c]
      b.atomicAdd_sync(addr,value, idx.d0)
  end
  atomicAddChannel:setname('Image.atomicAddChannel')

  return atomicAddChannel
end
-- IMAGE SPECIALIZATION END
------------------------------------------------------------------------------ wrap and compile kernels
-- TODO choose better name and put with other general purpose stuff
-- TODO this seems to be re-defined in solver AND o.t
local cd = macro(function(apicall) 
      return quote apicall end
    end)
b.cd = cd


-- TODO only used in next function, so make local there
local checkedLaunch = macro(function(kernelName, apicall)
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var name = [kernelName]
        var r = apicall
        if r ~= 0 then  
            C.printf("Kernel %s, Cuda reported error %d: %s\n", name, r, C.cudaGetErrorString(r))
            C.exit(r)
        end
    in
        r
    end end)

-- TODO only used with compiler functions below, group appropriately or find better way to inject this "global" variable
local GRID_SIZES = c.GRID_SIZES


-- TODO put in extra file for compiler stuff
local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel)
    kernelName = kernelName.."_"..tostring(ft)
    -- local kernelparams = compiledKernel:gettype().parameters
    -- local params = terralib.newlist {}
    -- for i = 3,#kernelparams do --skip GPU launcher and PlanData
    --     params:insert(symbol(kernelparams[i]))
    -- end
    -- print('asdf123')
    -- for k,v in pairs(params) do print(k,v) end
    -- error()
    -- local function createLaunchParameters(pd)
    --     if ft.kind == "CenteredFunction" then
    --         local ispace = ft.ispace
    --         local exps = terralib.newlist()
    --         for i = 1,3 do
    --            local dim = #ispace.dims >= i and ispace.dims[i].size or 1
    --             local bs = GRID_SIZES[#ispace.dims][i]
    --             exps:insert(dim)
    --             exps:insert(bs)
    --         end
    --         return exps
    --     else
    --         return {`pd.parameters.[ft.graphname].N,256,1,1,1,1}
    --     end
    -- end
    -- print('asdf123')
    -- print(compiledKernel)
    local terra GPULauncher(pd : &PlanData)
    --     var xdim,xblock,ydim,yblock,zdim,zblock = [ createLaunchParameters(pd) ]
            
    --     var launch = terralib.CUDAParams { (xdim - 1) / xblock + 1, (ydim - 1) / yblock + 1, (zdim - 1) / zblock + 1, 
    --                                         xblock, yblock, zblock, 
    --                                         0, nil }
    --     var stream : C.cudaStream_t = nil
    --     var endEvent : C.cudaEvent_t 
    --     if ([_opt_collect_kernel_timing]) then
    --         pd.timer:startEvent(kernelName,nil,&endEvent)
    --     end

    --     checkedLaunch(kernelName, compiledKernel(&launch, @pd, params))
        
    --     if ([_opt_collect_kernel_timing]) then
    --         pd.timer:endEvent(nil,endEvent)
    --     end

    --     cd(C.cudaGetLastError())

        var name = I.__itt_string_handle_create(kernelName)
        var domain = I.__itt_domain_create("Main.Domain")


        var endEvent : Event
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
        if ([_opt_collect_kernel_timing]) then
            pd.timer:startEvent(kernelName,&endEvent)
        end


        
      compiledKernel(@pd)

        if ([_opt_collect_kernel_timing]) then
            pd.timer:endEvent(&endEvent, 0)
        end
        I.__itt_task_end(domain)
    end
    -- print(GPULauncher)
    -- error()
    return GPULauncher
end


function b.makeWrappedFunctions(problemSpec, PlanData, delegate, names) -- same  problemSpec as in solver.t

    local function cpucompile(kernel, ispace)
      -- collect sizes of all dimensions associated with this ispace
      -- e.g. dimsizes = {100, 200} for a 100x200 picture
      local numdims = #(ispace.dims)
      local dims = ispace.dims
      local dimsizes = {}
      for k,dim in pairs(ispace.dims) do
        dimsizes[k] = dim.size
      end

      local pd_sym = symbol(PlanData, 'pd') -- symbol for the plandata
      local dimargs = terralib.newlist() -- symbols that represent the x,y,z coordinates
      for k = 1,numdims do
        dimargs:insert(symbol(int, 'x' .. tostring(k-1)))
      end

      local launchquote = quote 
        kernel([pd_sym], [dimargs])
      end

      local wrappedquote = launchquote


      for k = 1,numdims do
      -- for k = numdims,1,-1 do -- reverse loop order

        wrappedquote = quote

          for [dimargs[k]] = 0, [dimsizes[k]] do
            [wrappedquote]
          end

        end
      end
      print(wrappedquote)

      local wrappedfunc = terra([pd_sym])
        var name = I.__itt_string_handle_create('run_task_loop')
        var domain = I.__itt_domain_create("Main.Domain")


        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)

        escape
        local thequote = texec("compiledfunc(): loop time", false,
        -- local thequote = texec("compiledfunc(): loop time", true,
          wrappedquote
        )
        emit quote [thequote] end
        end

          -- [wrappedquote]

        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)
      end
      print(wrappedfunc)
      -- error()

      return wrappedfunc
    end

    -- print('\nInside util.makeGPUFunctions: The problemSpec.functions:')
    -- for k,v in pairs(problemSpec.functions) do print(k,v) end --debug 
    -- print('\nInside util.makeGPUFunctions: The problemSpec.functions[1]:')
    -- for k,v in pairs(problemSpec.functions[1]) do print(k,v) end --debug 
    -- print('\nInside util.makeGPUFunctions: The problemSpec.functions[1].typ:')
    -- for k,v in pairs(problemSpec.functions[1].typ) do print(k,v) end --debug 
    -- print('\nInside util.makeGPUFunctions: The problemSpec.functions[2]:')
    -- for k,v in pairs(problemSpec.functions[2]) do print(k,v) end --debug 
    -- print('\nInside util.makeGPUFunctions: The problemSpec.functions[2].typ:')
    -- for k,v in pairs(problemSpec.functions[2].typ) do print(k,v) end --debug 
    -- print('\nInside util.makeGPUFunctions: The problemSpec.functions[2].typ.__index:')
    -- for k,v in pairs(problemSpec.functions[2].typ.__index) do print(k,v) end --debug 
    -- print('\nInside util.makeGPUFunctions: The problemSpec:')
    -- for k,v in pairs(problemSpec.functions[1].typ.ispace.dims) do print(k,v) end
    -- for k,v in pairs(problemSpec.functions[1].typ.ispace.dims[1]) do print(k,v) end

    -- step 1: compile the actual cuda kernels
    local kernelFunctions = {}
    local key = tostring(os.time())
    local function getkname(name,ft)
        return string.format("%s_%s_%s",name,tostring(ft),key)
    end
    
    for _,problemfunction in ipairs(problemSpec.functions) do -- problemfunction is of type ProblemFunctions, see grammar in o.t
        if problemfunction.typ.kind == "CenteredFunction" then
           local ispace = problemfunction.typ.ispace
           local dimcount = #ispace.dims
               -- TODO this assert shouldn't be necessary for cpu backends.
	       assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
           local ks = delegate.CenterFunctions(ispace,problemfunction.functionmap) -- ks are the kernelfunctions as shown in gaussNewtonGPU.t
           for name,func in pairs(ks) do
             print(name,func) -- debug
                kernelFunctions[getkname(name,problemfunction.typ)] = cpucompile(func, ispace)
           end
            -- error()
        else
            local graphname = problemfunction.typ.graphname
            local ispace = problemfunction.typ.ispace -- by SO
            -- local ks = delegate.GraphFunctions(graphname,problemfunction.functionmap) -- original
            local ks = delegate.GraphFunctions(graphname, problemfunction.functionmap,nil, ispace) -- by SO
            for name,func in pairs(ks) do            
                kernelFunctions[getkname(name,problemfunction.typ)] = cpucompile(func, ispace)
                print(kernelFunctions[getkname(name,problemfunction.typ)])
            end
            -- error()
        end
    end
    -- error()
    
    -- print('\nIn makeGPUFunctions:') -- debug
    -- for k,v in pairs(kernelFunctions) do print(k,v) end
    local kernels = kernelFunctions
    -- print('\ncompiled cuda kernels')
    -- for k,v in pairs(kernels) do print(k,v) end -- end
    
    -- step 2: generate wrapper functions around each named thing
    local grouplaunchers = {}
    for _,name in ipairs(names) do -- name = e.g. 'PCGInit1', 'PCGStep1', etc.
        local args
        local launches = terralib.newlist()
        for _,problemfunction in ipairs(problemSpec.functions) do -- problemfunction is of type ProblemFunctions, see grammar in o.t
            local kname = getkname(name,problemfunction.typ)
            print(kname)
            local kernel = kernels[kname]
            if kernel then -- some domains do not have an associated kernel, (see _Finish kernels in GN which are only defined for 
                local launcher = makeGPULauncher(PlanData, name, problemfunction.typ, kernel)
                print(launcher) -- debug
                -- error()
                if not args then
                    args = launcher:gettype().parameters:map(symbol)
                end
                launches:insert(`launcher(args))
            else
                --print("not found: "..name.." for "..tostring(problemfunction.typ))
            end
        end
        local fn
        if not args then
            fn = macro(function() return `{} end) -- dummy function for blank groups occur for things like precompute and _Graph when they are not present
        else
            fn = terra([args]) [launches] end
            fn:setname(name)
            fn:gettype()
        end
        grouplaunchers[name] = fn 
    end

    print('\n\n\n')
    print('START inside backend.makeWrappedFunctions: the grouplaunchers')
    printt(grouplaunchers)
    print('END inside backend.makeWrappedFunctions: the grouplaunchers')
    print('\n\n\n')

    return grouplaunchers
end
-------------------------------- END wrap and compile kernels

return b
