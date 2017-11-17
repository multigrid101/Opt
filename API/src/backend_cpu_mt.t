local b = {}
local S = require("std")
local c = require('config')
local tp = require('threadpool')
local C = terralib.includecstring [[
#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]
local I = require('ittnotify')
local la = require('linalg_cpu_mt')

b.name = 'CPUMT'

local numthreads = c.numthreads
b.numthreads = numthreads

-- atomicAdd START
-- TODO make atomicAdd add into the sum, but make sure to take care of race conditions
-- OPTION 1: add into directly into global sum
-- OPTION 2: have each thread add into its own sum. (i.e. have 'sum' as a float[numthreads]) --> more efficient but harder to implement
-- b.summutex_sym = global(C.pthread_mutex_t[c.nummutexes], nil,  'summutex')
-- local tid_key = global(C.pthread_key_t, nil,  'tid_key')

b.threadarg = symbol(int, 'thread_id')
b.threadarg_val = b.threadarg -- need second variable to provide default arguments for other backends

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


--------------------------- Timing stuff start
-- TODO put in separate file
-- TODO what is this? (seems to be a lua "class" definition)
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
    if not T:isstruct() then
        terra Array:indexof(v : T) : int32
          escape
            emit quote
              for i = 0LL,self._size do
                  if C.strcmp(v,self._data[i])==0 then
                      return i
                  end
              end
              return -1
            end
          end
        end
        terra Array:contains(v : T) : bool
            return self:indexof(v) >= 0
        end
    end
	
    return Array
end

local Array = S.memoize(Array)


-- TODO what is this? its only used in the next few lines? can we make this local to the Timer "class"?
local struct Event {
	starttime : C.timespec
	endtime : C.timespec
	duration : double -- unit: ms
	eventName : rawstring
}
b.Event = Event
terra Event:getStartTime()
  var elapsed : double
  elapsed = 1000*(self.starttime.tv_sec)
  elapsed = elapsed + [double](self.starttime.tv_nsec)/([double](1e6))

  return elapsed
end
terra Event:getEndTime()
  var elapsed : double
  elapsed = 1000*(self.endtime.tv_sec)
  elapsed = elapsed + [double](self.endtime.tv_nsec)/([double](1e6))

  return elapsed
end
terra Event:calcElapsedTime()
  var elapsed : double

  -- elapsed = 1000*(self.endtime.tv_sec - self.starttime.tv_sec)
  -- elapsed = elapsed + [double](self.endtime.tv_usec - self.starttime.tv_usec)/([double](1e3))

  elapsed = 1000*(self.endtime.tv_sec - self.starttime.tv_sec)
  elapsed = elapsed + [double](self.endtime.tv_nsec - self.starttime.tv_nsec)/([double](1e6))

  self.duration = elapsed
end


-- one array for the main thread and one for each worker thread.
-- main thread uses index 0
-- worker thread with e.g. id=2 uses index 3
local struct Timer {
	eventList : (&Array(Event))[numthreads+1]
}
b.Timer = Timer




terra Timer:init() 
    for tid = 0,numthreads+1 do
	self.eventList[tid] = [Array(Event)].alloc():init()
    end
end

terra Timer:cleanup()
    for tid = 0,numthreads+1 do
	self.eventList[tid]:delete()
    end
end 


terra Timer:startEvent(name : rawstring, eventptr : &Event)
    (@eventptr).eventName = name
    -- C.gettimeofday(&((@eventptr).starttime), nil)
    C.clock_gettime(C.CLOCK_MONOTONIC, &((@eventptr).starttime))
end


terra Timer:endEvent(eventptr : &Event, [b.threadarg])
    -- C.gettimeofday(&((@eventptr).endtime), nil)
    C.clock_gettime(C.CLOCK_MONOTONIC, &((@eventptr).endtime))
    self.eventList[ [b.threadarg] ]:insert(@eventptr)
end

-- TODO only used in next function, so make local there
terra isprefix(pre : rawstring, str : rawstring) : bool
    if @pre == 0 then return true end
    if @str ~= @pre then return false end
    return isprefix(pre+1,str+1)
end
terra Timer:evaluate()
        -- put all worker-thread stuff into main-thread array
        for tid = 0,numthreads do
          for k = 0,self.eventList[tid+1]:size() do
            self.eventList[0]:insert(self.eventList[tid+1](k))
          end
        end

	-- if ([c._opt_verbosity > 0]) then
	if true then
          -- _0 holds duration
          -- _1 holds count
          -- _2 holds start
          -- _2 holds end
          var aggregateTimingInfo = [Array(tuple(float,int,float,float))].salloc():init()
          var aggregateTimingNames = [Array(rawstring)].salloc():init()

          for i = 0,self.eventList[0]:size() do
            var event = self.eventList[0](i);
            -- C.printf("%s\n", event.eventName)
            event:calcElapsedTime()

            -- if the event is not in the list, we insert it. if it is already in the list, we
            -- update the aggregate times and counts, but keep the start and end-time of the first
            -- recorded event
            var index =  aggregateTimingNames:indexof(event.eventName)
            if index < 0 then
              aggregateTimingNames:insert(event.eventName)
              aggregateTimingInfo:insert({event.duration, 1, event:getStartTime(), event:getEndTime()})
            else
              aggregateTimingInfo(index)._0 = aggregateTimingInfo(index)._0 + event.duration
              aggregateTimingInfo(index)._1 = aggregateTimingInfo(index)._1 + 1
            end
          end

          C.printf(		"---------------------------------------------------------------------------\n")
          C.printf(		"             Kernel             |   Count  |     Total     |      Average \n")
          C.printf(		"--------------------------------+----------+---------------+---------------\n")
          for i = 0, aggregateTimingNames:size() do
              C.printf(	"--------------------------------+----------+---------------+---------------\n")
              -- C.printf(" %-30s |   %4d   | %12.5fms| %12.5fms| %12.5fms| %12.5fms\n", aggregateTimingNames(i), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._2, aggregateTimingInfo(i)._3)
              C.printf(" %-30s |   %4d   | %12.5fms| %12.5fms\n", aggregateTimingNames(i), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1)
          end

          C.printf(		"---------------------------------------------------------------------------\n")
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

b.ReduceVar.getDataPtr = function(varquote, k)
  -- TODO the use of this function is confusing, (input = k, return = k+1)
  -- rename to e.g. getDataPtrForThreadID(var, tid), that would make more sense
  return `[varquote][k+1]
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

if c.opt_float == float then

    local terra atomicAdd_sync(sum : &float, value : float, offset : int)
      -- C.pthread_mutex_lock(&([b.summutex_sym][offset]))
      @sum = @sum + value
      -- C.pthread_mutex_unlock(&([b.summutex_sym][offset]))
    end
    local terra atomicAdd_nosync(sum : &float, value : float)
      @sum = @sum + value
    end
    b.atomicAdd_nosync = atomicAdd_nosync

    b.atomicAdd_sync = atomicAdd_sync
    -- TODO copy this implementation downwards when finished (or remove duplicates)
else
    struct ULLDouble {
        union {
            a : uint64;
            b : double;
        }
    }


    local terra __double_as_ull(v : double)
        var u : ULLDouble
        u.b = v;

        return u.a;
    end

    local terra __ull_as_double(v : uint64)
        var u : ULLDouble
        u.a = v;

        return u.b;
    end

    -- if pascalOrBetterGPU then
    if true then
        local terra atomicAdd_sync(sum : &double, value : double, offset: int)
          -- C.pthread_mutex_lock(&([b.summutex_sym][offset]))
          @sum = @sum + value
          -- C.pthread_mutex_unlock(&([b.summutex_sym][offset]))
        end
        local terra atomicAdd_nosync(sum : &float, value : float)
          @sum = @sum + value
        end
        b.atomicAdd_nosync = atomicAdd_nosync

        b.atomicAdd_sync = atomicAdd_sync
        -- b.atomicAdd_sync = atomicAdd_nosync
    else
        local terra atomicAdd_sync(sum : &double, value : double, offset : int)
          -- C.pthread_mutex_lock(&([b.summutex_sym][offset]))
          @sum = @sum + value
          -- C.pthread_mutex_unlock(&([b.summutex_sym][offset]))
        end
        local terra atomicAdd_nosync(sum : &float, value : float)
          @sum = @sum + value
        end
        b.atomicAdd_nosync = atomicAdd_nosync

        b.atomicAdd_sync = atomicAdd_sync
        -- b.atomicAdd_sync = atomicAdd_nosync
    end
end
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
-- TODO find out how we can avoid the extra helper arrays for pixel data, we only
-- need them for graph-data
-- --> it seems that atomicAdd is only used for graphs anyway, so at the moment we allocate
-- the extra arrays but they will not be used or traversed.
-- TODO perform initialization and summation on extra arrays in parallel
function b.make_Image_initGPU(imagetype_terra)
-- TODO change comments, they are not correct
-- we allocate (numthreads+1) arrays.
-- so for 2 threads, we have three arrays. the first one holds the actual values,
-- the other ones are temporary arrays used for summation etc.
-- example: 3 threads and array with 5 elements (for single threads)
-- | 0  | 1  | 2  | 3  | 4  | values
-- | 5  | 6  | 7  | 8  | 9  | thread 0 helper
-- | 10 | 11 | 12 | 13 | 14 | thread 1 helper
-- | 15 | 16 | 17 | 18 | 19 | thread 2 helper
--
-- if 'tid' is the thread id, and idx \in {0,1,2,3,4}, then the helpers can be
-- accessed via:
-- helper_idx = idx + length*(tid+1)
    local initGPU= terra(self : &imagetype_terra)
        var data : &uint8 -- we cast this to the correct type later inside setGPUptr
        -- C.printf('allocating space for %d arrays, with %d elements (%d bytes) each\n', numthreads+1, self:cardinality(), self:totalbytes())
        var helperData : &uint8
        
        -- TODO refactor
        -- b.cd( b.allocateDevice(&data, (numthreads+1)*self:totalbytes(), uint8) )
        -- b.cd( b.memsetDevice(data, 0, (numthreads+1)*self:totalbytes()) )
        b.cd( b.allocateDevice(&data, (1)*self:totalbytes(), uint8) )
        b.cd( b.memsetDevice(data, 0, (1)*self:totalbytes()) )

        b.cd( b.allocateDevice(&helperData, (numthreads)*self:totalbytes(), uint8) )
        b.cd( b.memsetDevice(helperData, 0, (numthreads)*self:totalbytes()) )


        self:initFromGPUptr(data) -- (short explanataion): set self.data = data (and cast to appropriate ptr-type)
        self:initHelperFromGPUptr(helperData) -- (short explanataion): set self.data = data (and cast to appropriate ptr-type)
    end
    initGPU:setname('Image.initGPU')

   return initGPU
end

function b.make_Image_metamethods__apply(imagetype_terra, indextype_terra, vectortype_terra, loadAsVector, VT)
-- TODO --> move this back to o.t
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
--TODO --> move this back to o.t
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
  local atomicAddChannel = terra(self : &imagetype_terra, idx : indextype_terra, c : int32, value : scalartype_terra, [b.threadarg])
      var tid = [b.threadarg]

      -- C.printf('    atomicAddChannel(): calling from thread %d, adding into index %d\n', tid, idx:tooffset() + self:cardinality()*(tid+1))

      -- TODO refactor
      -- var addr : &scalartype_terra = &self.data[idx:tooffset() + self:cardinality()*(tid+1)].data[c]
      var addr : &scalartype_terra = &self.helperData[idx:tooffset() + self:cardinality()*(tid)].data[c]
      b.atomicAdd_sync(addr,value, idx.d0)
  end
  atomicAddChannel:setname('Image.atomicAddChannel')

  return atomicAddChannel
end
-- IMAGE SPECIALIZATION END
------------------------------------------------------------------------------ wrap and compile kernels
local cd = macro(function(apicall) 
      return quote apicall end
    end)
b.cd = cd


-- local DEBUG_MUTEX = 1
local DEBUG_MUTEX = 0
local debm = macro(function(apicall)
if DEBUG_MUTEX == 1 then
  return quote apicall end
else
  return quote end
end
end)

-- checked pthread call
pth = {}
pth[16] = 'EBUSY'
for k,v in pairs(C) do print(k,v) end
-- local asdf = C.ESRCH

ptcode = global(int, 0, "ptcode")
local pt = macro(function(apicall)
  local apicallstr = tostring(apicall)
  return quote 
    var str = [apicallstr]
    ptcode = apicall
    if ptcode ~= 0 then
      C.printf('ERROR IN PTHREADS CALL, errorcode = %d\n', ptcode)
      C.printf('        in call: %s\n', str)
    end
  end
end)


local GRID_SIZES = c.GRID_SIZES


-- compiledKernel is the result of b.makeWrappedFunctions
local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel, ispace)
    kernelName = kernelName.."_"..tostring(ft)
-- Summary (assuming - for simplicity - that we are parallelizing a vector
-- addition with 4 worker threads):
-- 1) Define functions (taskfuncsAsLua) that will later be run by the worker
--    threads. each of these performs the vector addition for a quarter of the
--    indices.
-- 2) Define the function (GPULauncher) that will be executed by the main thread.
--    This function adds each of the functions from 1) to the taskQueue and
--    synchronizes afterwards
-- TODO generalize to arbitrary number of threads DONE
-- TODO current prevention of race-conditions in atomicAdd seems to be inefficient --> introduce separate sums for each thread DONE
-- TODO make sure that arrays are traversed in column-major order DONE
-- TODO make sure that granularity of thread-creation does not cause inefficiencies
    local numdims = #(ispace.dims)

    -- 1)
    local taskfuncsAsLua = {}
    for tid = 0,numthreads-1 do
      -- 1a) compute loop bounds
      local kmin = {}
      local kmax = {}

      -- outermost dimension is split among threads
      local dimsize_outerdim = ispace.dims[numdims].size
      local outerdim = numdims-1

      if tid < numthreads-1 then
        -- last thread needs to be set manually due to roundoff error
        kmin[ outerdim ] = tid*(dimsize_outerdim/numthreads)
        kmax[ outerdim ] = (tid+1)*(dimsize_outerdim/numthreads)
      else
        -- set outdim for final thread
        kmin[ outerdim ] = (numthreads-1)*(dimsize_outerdim/numthreads)
        kmax[ outerdim ] = dimsize_outerdim
      end

      -- all other dimensions traverse everything
      for d = 1,numdims-1 do
        local dimsize = ispace.dims[d].size
        kmin[ d-1 ] = 0
        kmax[ d-1 ] = dimsize
      end

      -- 1b) define the function for worker task
      taskfuncsAsLua[tid] = terra(arg : &opaque)
        -- import the loop limits from luacode above into terra code here
        debm( C.printf('starting taskfunc\n') )
        var domain = I.__itt_domain_create("Main.Domain")
        var name_boundaries = 
               I.__itt_string_handle_create('taskfunc(): preparing boundaries')
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_boundaries)

        var pd = [&PlanData](arg)
        var kmin_terra : int[numdims]
        var kmax_terra : int[numdims]

        escape
          for dim = 1,numdims do
            emit quote
              kmin_terra[ [dim-1] ] = [kmin[dim-1]]
              kmax_terra[ [dim-1] ] = [kmax[dim-1]]
            end
          end
        end
        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_boundaries)

        -- run loop within the thread-wise bounds
        var name_func = 
            I.__itt_string_handle_create('taskfunc(): running compiledKernel')
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_func)
        compiledKernel(@pd, kmin_terra, kmax_terra, [tid])
        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_func)

        tp.theKernelFinishedByAllThreadsBarrier:signal()

        debm( C.printf('stopping taskfunc\n') )

        var moreWorkWillCome = true
        return moreWorkWillCome
      end
    end
    for k,v in pairs(taskfuncsAsLua) do
      print('ASDFASDF'..kernelName)
      print(kernelName)
      print('ASDFASDF')
      print(k,v)
    end
    -- error()
        

    -- 2) define the function that is launched by the main thread
    local GPULauncher
    -- use single-threaded version if required
    if compiledKernel.compileForMultiThread == false then
      -- error()
      terra GPULauncher(pd : &PlanData)
          var name = I.__itt_string_handle_create(kernelName)
          var domain = I.__itt_domain_create("Main.Domain")


          var endEvent : Event
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent(kernelName, &endEvent)
          end

          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)

          -- I dont't need the values of these vars, they are just placeholders
          var tid = 0
          var kmin_terra : int[numdims]
          var kmax_terra : int[numdims]

          
          compiledKernel(@pd, kmin_terra, kmax_terra, tid)
          -- compiledKernel(@pd)

          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&endEvent, tid)
          end

          I.__itt_task_end(domain)
      end
    else -- compile for multi-thread
      terra GPULauncher(pd : &PlanData)
      -- C.sleep(1)
          debm( C.printf('starting GPULauncher\n') )
          var tasks : tp.Task_t[numthreads]

          -- import worker-thread-functions from lua into terra
          escape
            for k = 0,numthreads-1 do
              emit quote
                tasks[k] = tp.Task_t( { taskfunction=[ taskfuncsAsLua[k] ], arg=pd} )
              end
            end
          end


          var kernelEvent : Event 
          var threadStartEvent : Event 
          var helperArrayEvent : Event 

          var name = I.__itt_string_handle_create(kernelName)
          var domain = I.__itt_domain_create("Main.Domain")

          var name2 = I.__itt_string_handle_create('helperArrayTask')
          var name_waitTaskFinish = 
            I.__itt_string_handle_create('GPULauncher(): wait for tasks to finish')

          -- TODO find out if we need to optimize
          var eventKernelName = [&int8](C.malloc(100 * sizeof(int8)))
          C.sprintf(eventKernelName, "%s_total", kernelName)

          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent(eventKernelName, &kernelEvent)
          end

          -- set helper arrays to zero
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name2)
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent('helperArrayStuff',&helperArrayEvent)
          end
          escape
            for _,varname in pairs(compiledKernel.listOfAtomicAddVars) do
              print(varname)
              emit quote
                pd.[varname]:setHelperArraysToZero()
              end
            end
          end
          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&helperArrayEvent, 0)
          end
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name2)

          -- lock kernel_running_mutex
          tp.theKernelFinishedByAllThreadsBarrier:initialLock()

          -- make sure that the worker-threads are actually alive (via spinlock)
          -- we need this to ensure that taskQueue:set() (run by this, the main
          -- thread) doesn't signal worker threads that work is available before
          -- they are even alive (leads to deadlock).
          var waitForThreadsAliveName = 
            I.__itt_string_handle_create('wait_for_threads_to_start')
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent('waitThreadsStart',&threadStartEvent)
          end

          tp.theThreadsAliveBarrier:wait()

          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&threadStartEvent, 0)
          end
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)
         
          -- add tasks for workload to task-queue                                       
          for k = 0,numthreads do                                                     
            debm( C.printf('GPULauncher(): inserting task %d into taskQueue\n', k) )
            tp.theTaskQueue:set(k, tasks[k])                                                
          end                                                                         
          
          -- synchronize as next workload might depend on result of this workload 
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_waitTaskFinish)
          debm( C.printf('GPULauncher(): waiting for kernel-tasks to finish\n') )
          tp.theKernelFinishedByAllThreadsBarrier:wait()
          debm( C.printf('GPULauncher(): unlocking kernel_running_mutex\n') )
          tp.theKernelFinishedByAllThreadsBarrier:finalUnlock()
          debm( C.printf('inside GPULauncher3\n') )
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_waitTaskFinish)



          -- sum up the helper arrays
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name2)
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent('helperArrayStuff',&helperArrayEvent)
          end
          escape
            for _,varname in pairs(compiledKernel.listOfAtomicAddVars) do
              print(varname)
              emit quote
                pd.[varname]:sumUpHelperArrays()
              end
            end
          end
          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&helperArrayEvent, 0)
          end
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name2)
          -- TODO find out if we need to optimize

          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&kernelEvent, 0)
          end
          I.__itt_task_end(domain)

          debm( C.printf('stopping GPULauncher\n') )
      end
    end
    print(GPULauncher)
    -- error()
    return GPULauncher
end


function b.makeWrappedFunctions(problemSpec, PlanData, delegate, names) -- same  problemSpec as in solver.t

    local function cpucompile(kernel, ispace)
      -- collect sizes of all dimensions associated with this ispace
      -- e.g. dimsizes = {100, 200} for a 100x200 picture
      local numdims = #(ispace.dims)
      local kernelName = kernel.name
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

      local tidsym = symbol(int, 'tid')

      local launchquote = quote 
        kernel([pd_sym], [dimargs], [tidsym])
      end

      local kminsym = symbol(int[numdims], 'kmin')
      local kmaxsym = symbol(int[numdims], 'kmax')


      local wrappedquote = launchquote
      for k = 1,numdims do
        wrappedquote = quote

          for [dimargs[k]] = [kminsym][ [k-1] ], [kmaxsym][ [k-1] ] do
            [wrappedquote]
          end

        end
      end
      print(wrappedquote)

      -- for k,v in pairs(rawstring) do print(k,v) end
      -- error()

      local wrappedfunc = terra([pd_sym], [kminsym], [kmaxsym], [tidsym])

        var loopEvent : Event
        -- var loopEventName : &int8 = 'asdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf'
        var loopEventName = [&int8](C.malloc(50 * sizeof(int8)))
        C.sprintf(loopEventName, '%s_loop(%d)', kernelName, [tidsym])
        -- var thename = [rawstring](loopEventName)
        -- loopEventName = 'fdsa'

        var name = I.__itt_string_handle_create('run_task_loop')
        var domain = I.__itt_domain_create("Main.Domain")

        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
        if ([_opt_collect_kernel_timing]) then
            [pd_sym].timer:startEvent( (loopEventName) ,&loopEvent)
        end

        [wrappedquote]

        if ([_opt_collect_kernel_timing]) then
            [pd_sym].timer:endEvent(&loopEvent, [tidsym])
        end
        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)
        -- TODO generalize this to an arbitrary number of threads
        -- [pd_sym].scratch[0] = [pd_sym].scratch[0] + [pd_sym].scratch[1]
        -- [pd_sym].modelCost[0] = [pd_sym].modelCost[0] + [pd_sym].modelCost[1]
        -- [pd_sym].q[0] = [pd_sym].q[0] + [pd_sym].q[1]
        -- [pd_sym].scanAlphaDenominator[0] = [pd_sym].scanAlphaDenominator[0] + [pd_sym].scanAlphaDenominator[1]
        -- [pd_sym].scanAlphaNumerator[0] = [pd_sym].scanAlphaNumerator[0] + [pd_sym].scanAlphaNumerator[1]
        -- [pd_sym].scanBetaNumerator[0] = [pd_sym].scanBetaNumerator[0] + [pd_sym].scanBetaNumerator[1]
      end
      wrappedfunc:setname('wrappedfunc')
      wrappedfunc.listOfAtomicAddVars = kernel.listOfAtomicAddVars
      wrappedfunc.compileForMultiThread = true
      print(wrappedfunc)
      -- error()

      return wrappedfunc
    end



    local function cpucompile_forMainThread(kernel, ispace)
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

      local tidsym = symbol(int, 'tid')
      local launchquote = quote 
        kernel([pd_sym], [dimargs], [tidsym])
      end

      local wrappedquote = launchquote
      for k = 1,numdims do

          local l = numdims-(k-1)
        wrappedquote = quote

          for [dimargs[k]] = 0, [dimsizes[k]] do
          -- for [dimargs[l]] = 0, [dimsizes[l]] do
            [wrappedquote]
          end

        end
      end
      print(wrappedquote)

      -- local wrappedfunc = terra([pd_sym])
      local kminsym = symbol(int[numdims], 'kmin')
      local kmaxsym = symbol(int[numdims], 'kmax')
      local wrappedfunc = terra([pd_sym], [kminsym], [kmaxsym], [tidsym])
        var name = I.__itt_string_handle_create('run_task_loop')
        var domain = I.__itt_domain_create("Main.Domain")


        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)

        [wrappedquote]

        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)
      end
      print(wrappedfunc)
      -- error()

      wrappedfunc.compileForMultiThread = kernel.compileForMultiThread
      wrappedfunc.listOfAtomicAddVars = {} -- doesn't matter for this version

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
	       assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
           local ks = delegate.CenterFunctions(ispace,problemfunction.functionmap) -- ks are the kernelfunctions as shown in gaussNewtonGPU.t
           for name,func in pairs(ks) do
             -- print(name,func) -- debug
                local cpucompilefunc
                if func.compileForMultiThread == true then
                  cpucompilefunc = cpucompile
                elseif func.compileForMultiThread == false then
                  cpucompilefunc = cpucompile_forMainThread
                else
                  error('compileForMultiThread attribute of kernel not set')
                end
                func.name = name .. '_' ..  tostring(problemfunction.typ)
                kernelFunctions[getkname(name,problemfunction.typ)] = cpucompilefunc(func, ispace)
           end
        else
            local graphname = problemfunction.typ.graphname
            local ispace = problemfunction.typ.ispace -- by SO
            -- local ks = delegate.GraphFunctions(graphname,problemfunction.functionmap) -- original
            local ks = delegate.GraphFunctions(graphname, problemfunction.functionmap,nil, ispace) -- by SO
            for name,func in pairs(ks) do            

                local cpucompilefunc
                if func.compileForMultiThread == true then
                  cpucompilefunc = cpucompile
                elseif func.compileForMultiThread == false then
                  cpucompilefunc = cpucompile_forMainThread
                else
                  error('compileForMultiThread attribute of kernel not set')
                end

                -- func.name = name
                func.name = name .. '_' .. tostring(problemfunction.typ)
                kernelFunctions[getkname(name,problemfunction.typ)] = cpucompilefunc(func, ispace)
            end
        end
    end
    -- error()
    
    -- print('\nIn makeGPUFunctions:') -- debug
    -- for k,v in pairs(kernelFunctions) do print(k,v) end
    local kernels = kernelFunctions
    -- print('\ncompiled cuda kernels')
    for k,v in pairs(kernels) do print(k,v) end -- end
    -- error()
    
    -- step 2: generate wrapper functions around each named thing
    local grouplaunchers = {}
    for _,name in ipairs(names) do -- name = e.g. 'PCGInit1', 'PCGStep1', etc.
        local args
        local launches = terralib.newlist()
        for _,problemfunction in ipairs(problemSpec.functions) do -- problemfunction is of type ProblemFunctions, see grammar in o.t
            local ispace = problemfunction.typ.ispace
            local kname = getkname(name,problemfunction.typ)
            print(kname)
            local kernel = kernels[kname]
            if kernel then -- some domains do not have an associated kernel, (see _Finish kernels in GN which are only defined for 
                local launcher = makeGPULauncher(PlanData, name, problemfunction.typ, kernel, ispace)
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
    -- error()

    print('\n\n\n')
    print('START inside backend.makeWrappedFunctions: the grouplaunchers')
    printt(grouplaunchers)
    print('END inside backend.makeWrappedFunctions: the grouplaunchers')
    print('\n\n\n')
    -- error()

    return grouplaunchers
end
-------------------------------- END wrap and compile kernels

return b
