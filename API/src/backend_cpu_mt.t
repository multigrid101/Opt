local b = {}
local S = require("std")
local c = require('config')
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

b.name = 'CPUMT'

local numthreads = c.numthreads
b.numthreads = numthreads

-- atomicAdd START
-- TODO make atomicAdd add into the sum, but make sure to take care of race conditions
-- OPTION 1: add into directly into global sum
-- OPTION 2: have each thread add into its own sum. (i.e. have 'sum' as a float[numthreads]) --> more efficient but harder to implement
b.summutex_sym = global(C.pthread_mutex_t[c.nummutexes], nil,  'summutex')
local tid_key = global(C.pthread_key_t, nil,  'tid_key')

b.threadarg = symbol(int, 'thread_id')
b.threadarg_val = b.threadarg -- need second variable to provide default arguments for other backends
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
	-- starttime : C.timeval
	-- endtime : C.timeval
	starttime : C.timespec
	endtime : C.timespec
	duration : double -- unit: ms
	eventName : rawstring
}
b.Event = Event
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
          var aggregateTimingInfo = [Array(tuple(float,int))].salloc():init()
          var aggregateTimingNames = [Array(rawstring)].salloc():init()

          for i = 0,self.eventList[0]:size() do
            var event = self.eventList[0](i);
            C.printf("%s\n", event.eventName)
            event:calcElapsedTime()
            var index =  aggregateTimingNames:indexof(event.eventName)
            C.printf("index of event %s is %d\n", event.eventName, index)
            if index < 0 then
              aggregateTimingNames:insert(event.eventName)
              aggregateTimingInfo:insert({event.duration, 1})
            else
              aggregateTimingInfo(index)._0 = aggregateTimingInfo(index)._0 + event.duration
              aggregateTimingInfo(index)._1 = aggregateTimingInfo(index)._1 + 1
            end
          end

          C.printf(		"------------------------------------------------------------------\n")
          C.printf(		"             Kernel             |   Count  |   Total   | Average \n")
          C.printf(		"--------------------------------+----------+-----------+----------\n")
          for i = 0, aggregateTimingNames:size() do
              C.printf(	"--------------------------------+----------+-----------+----------\n")
              C.printf(" %-30s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames(i), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1)
          end

          C.printf(		"------------------------------------------------------------------\n")
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

        b.cd( b.allocateDevice(&data, (numthreads+1)*self:totalbytes(), uint8) )
        b.cd( b.memsetDevice(data, 0, (numthreads+1)*self:totalbytes()) )

        self:initFromGPUptr(data) -- (short explanataion): set self.data = data (and cast to appropriate ptr-type)
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

      var addr : &scalartype_terra = &self.data[idx:tooffset() + self:cardinality()*(tid+1)].data[c]
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

b.threadcreation_counter = global(int, 0,  'threadcreation_counter')

    -------------------------------- GLOBALS START
    theThreads = global(C.pthread_t[numthreads], nil, 'theThreads')

    thread_busy_mutex = global(C.pthread_mutex_t[numthreads], nil, "thread_busy_mutex")
    numkernels_finished_mutex = global(C.pthread_mutex_t, nil, "numkernels_finished_mutex")
    kernel_running_mutex = global(C.pthread_mutex_t, nil, "kernel_running_mutex")   
    thread_has_been_canceled_mutex = global(C.pthread_mutex_t, nil, "thread_has_been_canceled_mutex")   
    ready_for_work_mutex = global(C.pthread_mutex_t, nil, "ready_for_work_mutex")        
    numthreadsAliveMutex = global(C.pthread_mutex_t, nil, "numthreadsAlive")

    work_available_cv = global(C.pthread_cond_t[numthreads], nil, "work_available_cv")
    kernel_finished_cv = global(C.pthread_cond_t, nil, "kernel_finished_cv")        
    thread_has_been_canceled_cv = global(C.pthread_cond_t, nil, "thread_has_been_canceled_cv")        
    ready_for_work_cv = global(C.pthread_cond_t, nil, "ready_for_work_cv")        

    numkernels_finished = global(int, 0, "numkernels_finished")                     
    -- numwloads_finished = global(int, 0, "numwloads_finished")                      

    numthreadsAlive = global(int, 0, "numthreadsAlive")
    --------------------------------- GLOBALS END                  

    --------------------------------- Task_t START
    struct Task_t {
      taskfunction : {&opaque} -> {bool}
      pd : &opaque
    }

    terra Task_t:run() : bool
      debm( C.printf('Task_t:run(): starting\n') )
      debm( C.printf('Task_t:run(): self.taskfunction points to %d\n', self.taskfunction) )

      var name = I.__itt_string_handle_create('run_task')
      var domain = I.__itt_domain_create("Main.Domain")
      I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
      I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)


      var moreWorkWillCome = self.taskfunction(self.pd)
      debm( C.printf('Task_t:run(): stopping\n') )

      return moreWorkWillCome
    end
    --------------------------------- Task_t END

    --------------------------------- TaskQueue_t START
    TaskQueue_t = terralib.types.newstruct("TaskQueue_t")
    TaskQueue_t.entries:insert({ type = Task_t[numthreads], field = "threadTasks"})

    terra TaskQueue_t:get(threadIndex : int)
      return self.threadTasks[threadIndex]
    end                                                                                                                                                                                                                                            
    terra TaskQueue_t:set(threadIndex : int, task : Task_t)
      var domain = I.__itt_domain_create("Main.Domain")
      debm( C.printf('TaskQueue_t:set(): starting\n') )
      self.threadTasks[threadIndex] = task


      -- debm( C.printf('TaskQueue_t:set(): waiiting for readforwork signal\n') )
      --   C.pthread_mutex_lock(&ready_for_work_mutex)
      --   C.pthread_cond_wait(&ready_for_work_cv, &ready_for_work_mutex)
      --   C.pthread_mutex_unlock(&ready_for_work_mutex)



      debm( C.printf('TaskQueue_t:set(): starting to lock thread_busy_mutex[%d], its value is %d\n', threadIndex, thread_busy_mutex[0]) )
      var name_signal = I.__itt_string_handle_create('TaskQueue_t:set(): sending work signal')
      I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_signal)

      pt( C.pthread_mutex_lock(&thread_busy_mutex[threadIndex]))
      -- C.sleep(1)
      debm( C.printf('TaskQueue_t:set(): signaling that work is available for thread %d\n', threadIndex) )
      pt( C.pthread_cond_signal(&work_available_cv[threadIndex]))

      I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_signal)

      debm( C.printf('TaskQueue_t:set(): starting to unlock thread_busy_mutex[%d]\n', threadIndex) )
      pt( C.pthread_mutex_unlock(&thread_busy_mutex[threadIndex]))

      debm( C.printf('TaskQueue_t:set(): stopping\n') )
    end 
    taskQueue = global(TaskQueue_t, nil, "taskQueue")
    --------------------------------- TaskQueue_t END

local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel, ispace) -- compiledKernel is the result of b.makeWrappedFunctions





-- for k,v in pairs(compiledKernel) do print(k,v) end
-- print('ASDF')
-- printt(compiledKernel.listOfAtomicAddVars)
-- print('ASDF')
-- error()
    kernelName = kernelName.."_"..tostring(ft)
-- TODO generalize to arbitrary number of threads DONE
-- TODO current prevention of race-conditions in atomicAdd seems to be inefficient --> introduce separate sums for each thread DONE
-- TODO make sure that arrays are traversed in column-major order DONE
-- TODO make sure that granularity of thread-creation does not cause inefficiencies
    local numdims = #(ispace.dims)



    --------- THREAD POOL STUFF START




    local terra waitForWork(arg : &opaque) : &opaque                                      
      var threadIndex = [int64](arg)                                                
      debm( C.printf("waitForkWork(tid=%d): starting\n", threadIndex)                          )
      debm( C.printf("waitForkWork(tid=%d): locking thread_busy_mutex[%d], value before locking is %d\n", threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )
      pt( C.pthread_mutex_lock(&thread_busy_mutex[threadIndex])                         )
      debm( C.printf("waitForkWork(tid=%d): locking thread_busy_mutex[%d], value after locking is %d\n", threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )
      -- while numwloads_finished < NUMWLOADS  do                                   
        -- C.sleep(1)

      var moreWorkWillCome = true

        C.pthread_mutex_lock(&numthreadsAliveMutex)
        numthreadsAlive = numthreadsAlive + 1
        C.pthread_mutex_unlock(&numthreadsAliveMutex)

      while moreWorkWillCome  do                                                                
        debm( C.printf("waitForkWork(tid=%d): starting to wait for work, the value of thread_busy_mutex[%d] is %d\n", threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )

        -- debm( C.printf("waitForkWork(tid=%d): sending readyforwork signal\n", threadIndex, threadIndex, thread_busy_mutex[threadIndex]) )

        var name = I.__itt_string_handle_create('wait_for_work')
        var name2 = I.__itt_string_handle_create('pthread_exit')
        var domain = I.__itt_domain_create("Main.Domain")
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)

        pt( C.pthread_cond_wait(&work_available_cv[threadIndex], &thread_busy_mutex[threadIndex])) -- unlocks mutex on entering, locks it on exit)
        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)


        debm( C.printf("waitForkWork(tid=%d): after receiving signal for work, value of thread_busy_mutex is %d\n", threadIndex, thread_busy_mutex[threadIndex]) )

        debm( C.printf("waitForkWork(tid=%d): receiving work\n", threadIndex)                  )
        var task = taskQueue:get(threadIndex)                                       

        debm( C.printf("waitForkWork(tid=%d): running work\n", threadIndex)                    )
        moreWorkWillCome = task:run()                                                        
        debm( C.printf("waitForkWork(tid=%d): finished running work\n", threadIndex)                    )
      end                                                                           

      debm( C.printf("waitForkWork(tid=%d): unlocking thread_busy_mutex[%d]\n", threadIndex, threadIndex))
      pt( C.pthread_mutex_unlock(&thread_busy_mutex[threadIndex])                       )
      -- C.printf("waitForkWork(tid=%d): finished\n", threadIndex)                         

      -- I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name2)
      -- I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name2)
      debm( C.printf("waitForkWork(tid=%d): calling pthread_exit()\n", threadIndex, threadIndex))
      C.pthread_exit(nil)
      debm( C.printf("waitForkWork(tid=%d): calling pthread_exit()\n", threadIndex, threadIndex))
      return nil                                                                    
    end             

    local terra initGlobals()
      -- C.printf('initGlobals(): starting\n')
      -- C.pthread_key_create(&tid_key, nil)

      -- for k = 0,numthreads do                                                       
      --   C.printf('initGlobals(): value of thread_busy_mutex[%d] before init is %d\n', k, thread_busy_mutex[k])
      --   C.pthread_mutex_init(&thread_busy_mutex[k], nil)                            
      --   C.printf('initGlobals(): value of thread_busy_mutex[%d] after init is %d\n', k, thread_busy_mutex[k])
      --   C.pthread_cond_init(&work_available_cv[k], nil)                             
      -- end                                                                           
      -- C.pthread_mutex_init(&numkernels_finished_mutex, nil)                         
      -- C.pthread_mutex_init(&kernel_running_mutex, nil)                              
      -- C.pthread_cond_init(&kernel_finished_cv, nil)  
    end
    b.initGlobals = initGlobals

    -- TODO create corresponding join function and use in init, cost, and step()
    local terra initThreads()
      debm( C.printf('initThreads(): starting\n') )
      pt( C.pthread_key_create(&tid_key, nil))

      for k = 0,numthreads do                                                       
        debm( C.printf('initThreads(): value of thread_busy_mutex[%d] before init is %d\n', k, thread_busy_mutex[k]) )
        pt( C.pthread_mutex_init(&thread_busy_mutex[k], nil)                            )
        debm( C.printf('initThreads(): value of thread_busy_mutex[%d] after init is %d\n', k, thread_busy_mutex[k]) )
        pt( C.pthread_cond_init(&work_available_cv[k], nil)                             )
      end                                                                           
      pt( C.pthread_mutex_init(&thread_has_been_canceled_mutex, nil)                         )
      pt( C.pthread_mutex_init(&numkernels_finished_mutex, nil)                         )
      pt( C.pthread_mutex_init(&kernel_running_mutex, nil)                              )
      pt( C.pthread_mutex_init(&ready_for_work_mutex, nil)                              )
      pt( C.pthread_mutex_init(&numthreadsAliveMutex, nil)                              )

      pt( C.pthread_cond_init(&kernel_finished_cv, nil)  )
      pt( C.pthread_cond_init(&thread_has_been_canceled_cv, nil)  )
      pt( C.pthread_cond_init(&ready_for_work_cv, nil)  )

      numthreadsAlive = 0

      escape
        -- set cpu affinities
        if c.cpumap then
          emit quote
            var cpusets : C.cpu_set_t[numthreads]
            var cpumap : int[8]

            escape
              for k = 1,numthreads do
                emit quote
                  cpumap[ [k-1] ] = [ c.cpumap[k] ]
                end
              end
            end

            -- CPU_ZERO macro -- TODO refactor these macros
            for k = 0,numthreads do
              C.memset ( &(cpusets[k]) , 0, sizeof (C.cpu_set_t)) -- 0 is the integer value of '\0'
            end

            -- CPU_SET macro
            for k = 0,numthreads do
              var cpuid : C.size_t = cpumap[k]
              ([&C.__cpu_mask](cpusets[k].__bits))[0] = ([&C.__cpu_mask](cpusets[k].__bits))[0] or ([C.__cpu_mask]( 1  << cpuid) )
            end


            for k = 0,numthreads do
              tdatas[k].cpuset = cpusets[k]
            end
          end
        end
      end

      for tid = 0,numthreads do
        pt( C.pthread_create(&theThreads[tid], nil, waitForWork, [&opaque](tid)))
      end
      -- C.sleep(1)
      debm( C.printf('initThreads(): stopping\n') )
    end
    b.initThreads = initThreads

    local terra stopWaitingForWork(dummy : &opaque)
       -- C.printf('stopWaitingForWork(): starting\n')
       -- var thisThread = C.pthread_self()

       -- C.printf('stopWaitingForWork(): canceling thread\n')
       -- pt( C.pthread_cancel(thisThread) )

      -- pt( C.pthread_mutex_lock(&thread_has_been_canceled_mutex))
      -- pt( C.pthread_cond_signal(&thread_has_been_canceled_cv))
      -- pt( C.pthread_mutex_unlock(&thread_has_been_canceled_mutex))

       -- C.printf('stopWaitingForWork(): stopping\n')

       var moreWorkWillCome = false
       return moreWorkWillCome
    end

    local stopWaitingForWorkTask = global(Task_t, `Task_t( {taskfunction=stopWaitingForWork, pd = nil} ), 'stopWaitingForWorkTask')

    local terra joinThreads()
      debm( C.printf('joinThreads(): starting\n') )

        -- wait for all threads to start
        var waitForThreadsAliveName = I.__itt_string_handle_create('wait_for_threads_to_start')
        var domain = I.__itt_domain_create("Main.Domain")
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)
        while true do
          C.pthread_mutex_lock(&numthreadsAliveMutex)
          var numalive = numthreadsAlive
          C.pthread_mutex_unlock(&numthreadsAliveMutex)

          if numalive == numthreads then
            break
          end
        end
        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)

      for tid = 0,numthreads do                                                       
        taskQueue:set(tid, stopWaitingForWorkTask)
      end

      -- pt( C.pthread_mutex_lock(&thread_has_been_canceled_mutex))
      -- pt( C.pthread_cond_wait(&thread_has_been_canceled_cv, &thread_has_been_canceled_mutex))
      -- pt( C.pthread_mutex_unlock(&thread_has_been_canceled_mutex))


      debm( C.printf('joinThreads(): waiting for threads to join\n') )
      for tid = 0,numthreads do                                                       
        pt( C.pthread_join(theThreads[tid], nil))
        -- pt( C.pthread_cancel(theThreads[tid]))
      end
      debm( C.printf('joinThreads(): threads are joined\n') )

      -- pt( C.pthread_key_create(&tid_key, nil))

      for k = 0,numthreads do                                                       
        pt( C.pthread_mutex_destroy(&thread_busy_mutex[k])                            )
        pt( C.pthread_cond_destroy(&work_available_cv[k]))
      end                                                                           
      debm( C.printf('joinThreads(): bla1\n') )
      pt( C.pthread_mutex_destroy(&numkernels_finished_mutex))
      pt( C.pthread_mutex_destroy(&kernel_running_mutex)                              )
      pt( C.pthread_cond_destroy(&kernel_finished_cv)  )
      debm( C.printf('joinThreads(): bla2\n') )

      pt( C.pthread_cond_destroy(&thread_has_been_canceled_cv)  )
      pt( C.pthread_mutex_destroy(&thread_has_been_canceled_mutex) )
      debm( C.printf('joinThreads(): bla3\n') )

      pt( C.pthread_mutex_destroy(&numthreadsAliveMutex) )

      -- pt( C.pthread_cond_destroy(&thread_has_been_canceled_cv)  )
      -- pt( C.pthread_mutex_destroy(&thread_has_been_canceled_mutex) )


      for k = 0,numthreads do
        -- C.printf('joinThreads(): unlocking thread_busy_mutex[%d], its value is %d\n', k, thread_busy_mutex[k])
        -- pt( C.pthread_mutex_unlock(&thread_busy_mutex[k]))
        -- C.printf('joinThreads(): after unlocking thread_busy_mutex[%d] its value is %d\n', k, thread_busy_mutex[k])
        -- pt( C.pthread_cancel(theThreads[k]) ) --> moved to joinThreads)
      end
      debm( C.printf('joinThreads(): stopping\n') )
    end
    b.joinThreads = joinThreads

    local taskfuncsAsLua = {}
    for tid = 0,numthreads-1 do
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

      taskfuncsAsLua[tid] = terra(arg : &opaque)
        var domain = I.__itt_domain_create("Main.Domain")

        var name_boundaries = I.__itt_string_handle_create('taskfunc(): preparing boundaries')
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_boundaries)


        debm( C.printf('starting taskfunc\n') )
        var pd = [&PlanData](arg)
        var kmin_terra : int[numdims]
        var kmax_terra : int[numdims]
        debm( C.printf('inside taskfunc1\n') )

        escape
          for dim = 1,numdims do
            emit quote
              kmin_terra[ [dim-1] ] = [kmin[dim-1]]
              kmax_terra[ [dim-1] ] = [kmax[dim-1]]
            end
          end
        end

        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_boundaries)

        var name_func = I.__itt_string_handle_create('taskfunc(): running compiledKernel')
        I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_func)
        compiledKernel(@pd, kmin_terra, kmax_terra, [tid])
        I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_func)

        pt( C.pthread_mutex_lock(&numkernels_finished_mutex)                            )
        debm( C.printf("taskfun(): increasing numkernels_finished counter\n")              )
        numkernels_finished = numkernels_finished + 1                               
         
        debm( C.printf("taskfun(): checking if all threads are done\n")                    )
        if numkernels_finished == numthreads then                                   

          var name = I.__itt_string_handle_create('taskfunc(): sending kernel_finished signal')
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
          pt( C.pthread_mutex_lock(&kernel_running_mutex)                               )
          pt( C.pthread_cond_signal(&kernel_finished_cv)                                )
          pt( C.pthread_mutex_unlock(&kernel_running_mutex)                             )
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name)
        end                                                                         
        pt( C.pthread_mutex_unlock(&numkernels_finished_mutex)                          )
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
    -- THREADPOOL END




    -- local struct thread_data {
    --   kmin : int[numdims],
    --   kmax : int[numdims],
    --   pd : &PlanData
    --   tid : int -- thread id
    --   cpuset : C.cpu_set_t
    -- }

    -- local terra threadLauncher(threadarg : &opaque) : &opaque
    --     var threaddata = [&thread_data](threadarg)
    --     var pd = threaddata.pd
    --     var kmin = threaddata.kmin
    --     var kmax = threaddata.kmax
    --     var tid = threaddata.tid
    --     C.pthread_setspecific(tid_key, [&opaque](tid-1))

    --     var cpuset = threaddata.cpuset
        
    --     -- if config.cpumap is not set, then let OS schedule the threads as it sees fit
    --     escape
    --       if c.cpumap then
    --         emit quote
    --           C.pthread_setaffinity_np(C.pthread_self(), sizeof(C.cpu_set_t), &cpuset)
    --         end
    --       end
    --     end

    --     compiledKernel(@pd, kmin, kmax, tid)
        

    -- end
    local GPULauncher
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
    else
      terra GPULauncher(pd : &PlanData)
      -- C.sleep(1)
          debm( C.printf('starting GPULauncher\n') )
          -- TODO THREADPOOL START: this function needs to add tasks to the task-queue
          var tasks : Task_t[numthreads]

          escape
            for k = 0,numthreads-1 do
              emit quote
                tasks[k] = Task_t( { taskfunction=[ taskfuncsAsLua[k] ], pd=pd} )
              end
            end
          end


          var endEvent : Event 
          var kernelEvent : Event 
          var threadStartEvent : Event 
          var helperArrayEvent : Event 

          var name = I.__itt_string_handle_create(kernelName)
          var domain = I.__itt_domain_create("Main.Domain")

          var name2 = I.__itt_string_handle_create('helperArrayTask')
          var name_waitTaskFinish = I.__itt_string_handle_create('GPULauncher(): wait for tasks to finish')

          -- timer start
          -- TODO find out if we need to optimize
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent(kernelName,&endEvent)
          end

          -- REDUCEVECTOR INIT
          -- pd:setHelperArraysToZero()
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

          debm( C.printf('GPULauncher(): locking kernel_running_mutex\n') )
          pt( C.pthread_mutex_lock(&kernel_running_mutex)                                 )
          
          debm( C.printf('GPULauncher(): locking numkernels_finished_mutex\n') )
          pt( C.pthread_mutex_lock(&numkernels_finished_mutex)                            )
          debm( C.printf('GPULauncher(): setting numkernels_finished to zero\n') )
          numkernels_finished = 0                                                     
          debm( C.printf('GPULauncher(): unlocking numkernels_finished_mutex\n') )
          pt( C.pthread_mutex_unlock(&numkernels_finished_mutex)                          )

          -- wait for all threads to start
          var waitForThreadsAliveName = I.__itt_string_handle_create('wait_for_threads_to_start')
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent('waitThreadsStart',&threadStartEvent)
          end
          while true do
            C.pthread_mutex_lock(&numthreadsAliveMutex)
            var numalive = numthreadsAlive
            C.pthread_mutex_unlock(&numthreadsAliveMutex)

            if numalive == numthreads then
              break
            end
          end
          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&threadStartEvent, 0)
          end
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, waitForThreadsAliveName)
         
          -- add tasks for workload to task-queue                                       
          for k = 0,numthreads do                                                     
            debm( C.printf('GPULauncher(): inserting task %d into taskQueue\n', k) )
            taskQueue:set(k, tasks[k])                                                
          end                                                                         
          
          -- synchronize as next workload might depend on result of previous workload 
          I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_waitTaskFinish)
          debm( C.printf('GPULauncher(): waiting for kernel-tasks to finish\n') )
          pt( C.pthread_cond_wait(&kernel_finished_cv, &kernel_running_mutex)             )
          debm( C.printf('GPULauncher(): unlocking kernel_running_mutex\n') )
          pt( C.pthread_mutex_unlock(&kernel_running_mutex))
          debm( C.printf('inside GPULauncher3\n') )
          I.__itt_task_end(domain, I.__itt_null, I.__itt_null, name_waitTaskFinish)



          -- REDUCEVECTOR SUM UP
          -- pd:sumUpHelperArrays()
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

          -- timer stop
          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(&endEvent, 0)
          end
          I.__itt_task_end(domain)


          -- THREADPOOL END

          -- var [b.summutex_sym]



          -- var tdata1 : thread_data
          -- var tdata2 : thread_data

          -- var tdatas : thread_data[numthreads] --> no longer necessary

          -- var t1 : C.pthread_t
          -- var t2 : C.pthread_t

          -- var threads : C.pthread_t[numthreads] --> now a global var

          -- C.pthread_key_create(&tid_key, nil) --> moved to initThreads()

          -- if config.cpumap is not set, then let OS take care of threadmapping
          -- escape --> moved to initThreads()
          --   -- set cpu affinities
          --   if c.cpumap then
          --     emit quote
          --       var cpusets : C.cpu_set_t[numthreads]
          --       var cpumap : int[8]

          --       escape
          --         for k = 1,numthreads do
          --           emit quote
          --             cpumap[ [k-1] ] = [ c.cpumap[k] ]
          --           end
          --         end
          --       end

          --       -- CPU_ZERO macro -- TODO refactor these macros
          --       for k = 0,numthreads do
          --         C.memset ( &(cpusets[k]) , 0, sizeof (C.cpu_set_t)) -- 0 is the integer value of '\0'
          --       end

          --       -- CPU_SET macro
          --       for k = 0,numthreads do
          --         var cpuid : C.size_t = cpumap[k]
          --         ([&C.__cpu_mask](cpusets[k].__bits))[0] = ([&C.__cpu_mask](cpusets[k].__bits))[0] or ([C.__cpu_mask]( 1  << cpuid) )
          --       end


          --       for k = 0,numthreads do
          --         tdatas[k].cpuset = cpusets[k]
          --       end
          --     end
          --   end
          -- end

          -- -- KMIN/KMAX CALCULATION START
          -- -- TODO balance workload more evenly (if necessary)
          -- -- set threadData values, i.e. kmin, kmax, etc.
          -- escape --> now calculated in taskfunctions
          --     -- outermost dimension is split among threads
          --     local dimsize = ispace.dims[numdims].size
          --     local outerdim = numdims-1
          --     -- local outerdim = 0
          --     emit quote
          --       -- tdata1.kmin[ 0 ] = 0
          --       -- tdata1.kmax[ 0 ] = dimsize/2

          --       -- tdata2.kmin[ 0 ] = dimsize/2
          --       -- tdata2.kmax[ 0 ] = dimsize
          --       for k = 0,numthreads-1 do -- last thread needs to be set manually due to roundoff error
          --         tdatas[k].kmin[ outerdim ] = k*(dimsize/numthreads)
          --         tdatas[k].kmax[ outerdim ] = (k+1)*(dimsize/numthreads)
          --       end

          --       tdatas[numthreads-1].kmin[ outerdim ] = (numthreads-1)*(dimsize/numthreads)
          --       tdatas[numthreads-1].kmax[ outerdim ] = dimsize
          --     end

          --   -- all other dimensions traverse everything
          --   -- for d = 2,numdims do
          --   for d = 1,numdims-1 do
          --     -- local dimsize = ispace.dims[numdims-d].size
          --     local dimsize = ispace.dims[d].size
          --     emit quote
          --       for k = 0,numthreads do
          --         -- tdata1.kmin[ [d-1] ] = 0
          --         -- tdata1.kmax[ [d-1] ] = dimsize

          --         -- tdata2.kmin[ [d-1] ] = 0
          --         -- tdata2.kmax[ [d-1] ] = dimsize
          --         tdatas[k].kmin[ [d-1] ] = 0
          --         tdatas[k].kmax[ [d-1] ] = dimsize
          --       end
          --     end
          --   end
          -- end

          -- -- tdata1.pd = pd
          -- -- tdata2.pd = pd

          -- -- tdata1.tid = 1
          -- -- tdata2.tid = 2
          -- for k = 0,numthreads do
          --   tdatas[k].pd = pd
          --   tdatas[k].tid = k+1
          -- end
          -- -- KMIN/KMAX CALCULATION END

          -- C.pthread_create(&t1, nil, threadLauncher, &tdata1)
          -- C.pthread_create(&t2, nil, threadLauncher, &tdata2)

          -- following block moved to GPULauncher
          -- var endEvent : C.cudaEvent_t 
          -- var threadEvent : C.cudaEvent_t 
          -- var kernelEvent : C.cudaEvent_t 
          -- if ([_opt_collect_kernel_timing]) then
          --     pd.timer:startEvent(kernelName,nil,&endEvent)
          -- end

          -- if ([_opt_collect_kernel_timing]) then
          --     pd.timer:startEvent('kernel',nil,&kernelEvent)
          -- end

          -- following block moved to GPULauncher
          -- var name = I.__itt_string_handle_create(kernelName)
          -- var domain = I.__itt_domain_create("Main.Domain")

          -- following block moved to GPULauncher
          -- I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)

          -- -- REDUCEVECTOR INIT --> moved to GPULauncher()
          -- -- pd:setHelperArraysToZero()
          -- escape
          --   for _,varname in pairs(compiledKernel.listOfAtomicAddVars) do
          --     print(varname)
          --     emit quote
          --       pd.[varname]:setHelperArraysToZero()
          --     end
          --   end
          -- end
          -- -- TODO find out if we need to optimize

          

          for k = 0,numthreads do
            -- following block no longer necessary
            -- if ([_opt_collect_kernel_timing]) then
            --     pd.timer:startEvent('thread_start',nil,&threadEvent)
            -- end

            -- [b.threadcreation_counter] = [b.threadcreation_counter] + 1 --> no longer necessary
            -- C.pthread_create(&threads[k], nil, threadLauncher, &tdatas[k]) --> replaced by TaskQueue:set()

            -- following block no longer necessary
            -- if ([_opt_collect_kernel_timing]) then
            --     pd.timer:endEvent(nil,threadEvent)
            -- end
          end
          

          -- C.pthread_join(t1, nil)
          -- C.pthread_join(t2, nil)
          for k = 0,numthreads do
            -- if ([_opt_collect_kernel_timing]) then
            --     pd.timer:startEvent('thread_start',nil,&threadEvent)
            -- end

            -- C.pthread_join(threads[k], nil) --> moved to joinThreads

            -- if ([_opt_collect_kernel_timing]) then
            --     pd.timer:endEvent(nil,threadEvent)
            -- end
          end

          -- -- REDUCEVECTOR SUM UP
          -- -- pd:sumUpHelperArrays()
          -- escape --> moved to GPULauncher()
          --   for _,varname in pairs(compiledKernel.listOfAtomicAddVars) do
          --     print(varname)
          --     emit quote
          --       pd.[varname]:sumUpHelperArrays()
          --     end
          --   end
          -- end
          -- -- TODO find out if we need to optimize

          -- if ([_opt_collect_kernel_timing]) then
          --     pd.timer:endEvent(nil,kernelEvent)
          -- end

          -- following block was moved to GPULaucher
          -- if ([_opt_collect_kernel_timing]) then
          --     pd.timer:endEvent(nil,endEvent)
          -- end


          -- following block was moved to GPULaucher
          -- I.__itt_task_end(domain)


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
