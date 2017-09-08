local b = {}
local c = require('config')
local C = terralib.includecstring [[
#define _GNU_SOURCE
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]

b.name = 'CPUMT'

local numthreads = c.numthreads
b.numthreads = numthreads

-- atomicAdd START
-- TODO make atomicAdd add into the sum, but make sure to take care of race conditions
-- OPTION 1: add into directly into global sum
-- OPTION 2: have each thread add into its own sum. (i.e. have 'sum' as a float[numthreads]) --> more efficient but harder to implement
b.summutex_sym = global(C.pthread_mutex_t[c.nummutexes], nil,  'summutex')

b.threadarg = symbol(int, 'thread_id')
b.threadarg_val = b.threadarg -- need second variable to provide default arguments for other backends

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

if c.opt_float == float then

    local terra atomicAdd_sync(sum : &float, value : float, offset : int)
      C.pthread_mutex_lock(&([b.summutex_sym][offset]))
      @sum = @sum + value
      C.pthread_mutex_unlock(&([b.summutex_sym][offset]))
    end
    local terra atomicAdd_nosync(sum : &float, value : float)
      @sum = @sum + value
    end
    b.atomicAdd_nosync = atomicAdd_nosync

    b.atomicAdd_sync = atomicAdd_sync
    -- b.atomicAdd_sync = atomicAdd_nosync
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
          C.pthread_mutex_lock(&([b.summutex_sym][offset]))
          @sum = @sum + value
          C.pthread_mutex_unlock(&([b.summutex_sym][offset]))
        end
        local terra atomicAdd_nosync(sum : &float, value : float)
          @sum = @sum + value
        end
        b.atomicAdd_nosync = atomicAdd_nosync

        b.atomicAdd_sync = atomicAdd_sync
        -- b.atomicAdd_sync = atomicAdd_nosync
    else
        local terra atomicAdd_sync(sum : &double, value : double, offset : int)
          C.pthread_mutex_lock(&([b.summutex_sym][offset]))
          @sum = @sum + value
          C.pthread_mutex_unlock(&([b.summutex_sym][offset]))
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

------------------------------------------------------------------------------ wrap and compile kernels
local cd = macro(function(apicall) 
      return quote apicall end
    end)
b.cd = cd



local GRID_SIZES = c.GRID_SIZES

b.threadcreation_counter = global(int, 0,  'threadcreation_counter')


local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel, ispace) -- compiledKernel is the result of b.makeWrappedFunctions
-- TODO generalize to arbitrary number of threads DONE
-- TODO current prevention of race-conditions in atomicAdd seems to be inefficient --> introduce separate sums for each thread
-- TODO make sure that arrays are traversed in column-major order DONE
-- TODO make sure that granularity of thread-creation does not cause inefficiencies
    local numdims = #(ispace.dims)

    local struct thread_data {
      kmin : int[numdims],
      kmax : int[numdims],
      pd : &PlanData
      tid : int -- thread id
      cpuset : C.cpu_set_t
    }

    local terra threadLauncher(threadarg : &opaque) : &opaque
        var threaddata = [&thread_data](threadarg)
        var pd = threaddata.pd
        var kmin = threaddata.kmin
        var kmax = threaddata.kmax
        var tid = threaddata.tid

        var cpuset = threaddata.cpuset
        
        -- if config.cpumap is not set, then let OS schedule the threads as it sees fit
        escape
          if c.cpumap then
            emit quote
              C.pthread_setaffinity_np(C.pthread_self(), sizeof(C.cpu_set_t), &cpuset)
            end
          end
        end

        compiledKernel(@pd, kmin, kmax, tid)
        

    end
-- 
    local terra GPULauncher(pd : &PlanData)
        -- var [b.summutex_sym]


        -- var tdata1 : thread_data
        -- var tdata2 : thread_data

        var tdatas : thread_data[numthreads]

        -- var t1 : C.pthread_t
        -- var t2 : C.pthread_t

        var threads : C.pthread_t[numthreads]

        -- if config.cpumap is not set, then let OS take care of threadmapping
        escape
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

              -- CPU_ZERO macro
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

        -- TODO this is not the correct way to split up the work, will only work for one dimension.
        -- TODO balance workload more evenly (if necessary)
        escape
            -- outermost dimension is split among threads
            local dimsize = ispace.dims[1].size
            local outerdim = numdims-1
            -- local outerdim = 0
            emit quote
              -- tdata1.kmin[ 0 ] = 0
              -- tdata1.kmax[ 0 ] = dimsize/2

              -- tdata2.kmin[ 0 ] = dimsize/2
              -- tdata2.kmax[ 0 ] = dimsize
              for k = 0,numthreads-1 do -- last thread needs to be set manually due to roundoff error
                tdatas[k].kmin[ outerdim ] = k*(dimsize/numthreads)
                tdatas[k].kmax[ outerdim ] = (k+1)*(dimsize/numthreads)
              end

              tdatas[numthreads-1].kmin[ outerdim ] = (numthreads-1)*(dimsize/numthreads)
              tdatas[numthreads-1].kmax[ outerdim ] = dimsize
            end

          -- all other dimensions traverse everything
          -- for d = 2,numdims do
          for d = 1,numdims-1 do
            local dimsize = ispace.dims[d].size
            emit quote
              for k = 0,numthreads do
                -- tdata1.kmin[ [d-1] ] = 0
                -- tdata1.kmax[ [d-1] ] = dimsize

                -- tdata2.kmin[ [d-1] ] = 0
                -- tdata2.kmax[ [d-1] ] = dimsize
                tdatas[k].kmin[ [d-1] ] = 0
                tdatas[k].kmax[ [d-1] ] = dimsize
              end
            end
          end
        end

        -- tdata1.pd = pd
        -- tdata2.pd = pd

        -- tdata1.tid = 1
        -- tdata2.tid = 2
        for k = 0,numthreads do
          tdatas[k].pd = pd
          tdatas[k].tid = k+1
        end

        -- C.pthread_create(&t1, nil, threadLauncher, &tdata1)
        -- C.pthread_create(&t2, nil, threadLauncher, &tdata2)
        var endEvent : C.cudaEvent_t 
        var threadEvent : C.cudaEvent_t 
        if ([_opt_collect_kernel_timing]) then
            pd.timer:startEvent(kernelName,nil,&endEvent)
        end

        

        for k = 0,numthreads do
          if ([_opt_collect_kernel_timing]) then
              pd.timer:startEvent('thread start',nil,&threadEvent)
          end

          [b.threadcreation_counter] = [b.threadcreation_counter] + 1
          C.pthread_create(&threads[k], nil, threadLauncher, &tdatas[k])

          if ([_opt_collect_kernel_timing]) then
              pd.timer:endEvent(nil,threadEvent)
          end
        end
        

        -- C.pthread_join(t1, nil)
        -- C.pthread_join(t2, nil)
        for k = 0,numthreads do

          C.pthread_join(threads[k], nil)

        end

        if ([_opt_collect_kernel_timing]) then
            pd.timer:endEvent(nil,endEvent)
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

      local wrappedfunc = terra([pd_sym], [kminsym], [kmaxsym], [tidsym])
        [wrappedquote]
        -- TODO generalize this to an arbitrary number of threads
        -- [pd_sym].scratch[0] = [pd_sym].scratch[0] + [pd_sym].scratch[1]
        -- [pd_sym].modelCost[0] = [pd_sym].modelCost[0] + [pd_sym].modelCost[1]
        -- [pd_sym].q[0] = [pd_sym].q[0] + [pd_sym].q[1]
        -- [pd_sym].scanAlphaDenominator[0] = [pd_sym].scanAlphaDenominator[0] + [pd_sym].scanAlphaDenominator[1]
        -- [pd_sym].scanAlphaNumerator[0] = [pd_sym].scanAlphaNumerator[0] + [pd_sym].scanAlphaNumerator[1]
        -- [pd_sym].scanBetaNumerator[0] = [pd_sym].scanBetaNumerator[0] + [pd_sym].scanBetaNumerator[1]
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
	       assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
           local ks = delegate.CenterFunctions(ispace,problemfunction.functionmap) -- ks are the kernelfunctions as shown in gaussNewtonGPU.t
           for name,func in pairs(ks) do
             -- print(name,func) -- debug
                kernelFunctions[getkname(name,problemfunction.typ)] = cpucompile(func, ispace)
           end
        else
            local graphname = problemfunction.typ.graphname
            local ispace = problemfunction.typ.ispace -- by SO
            -- local ks = delegate.GraphFunctions(graphname,problemfunction.functionmap) -- original
            local ks = delegate.GraphFunctions(graphname, problemfunction.functionmap,nil, ispace) -- by SO
            for name,func in pairs(ks) do            
                kernelFunctions[getkname(name,problemfunction.typ)] = cpucompile(func, ispace)
            end
        end
    end
    
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

    print('\n\n\n')
    print('START inside backend.makeWrappedFunctions: the grouplaunchers')
    printt(grouplaunchers)
    print('END inside backend.makeWrappedFunctions: the grouplaunchers')
    print('\n\n\n')

    return grouplaunchers
end
-------------------------------- END wrap and compile kernels

return b
