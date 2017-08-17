local b = {}
local c = require('config')
local C = terralib.includecstring [[
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

-- atomicAdd START
-- TODO make atomicAdd add into the sum, but make sure to take care of race conditions
-- OPTION 1: add into directly into global sum
-- OPTION 2: have each thread add into its own sum. (i.e. have 'sum' as a float[numthreads]) --> more efficient but harder to implement
b.summutex_sym = global(C.pthread_mutex_t, nil,  'summutex')
if c.opt_float == float then

    local terra atomicAdd_sync(sum : &float, value : float)
      C.pthread_mutex_lock(&[b.summutex_sym])
      @sum = @sum + value
      C.pthread_mutex_unlock(&[b.summutex_sym])
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

    if pascalOrBetterGPU then
        local terra atomicAdd_sync(sum : &double, value : double)
          C.pthread_mutex_lock(&[b.summutex_sym])
          @sum = @sum + value
          C.pthread_mutex_unlock(&[b.summutex_sym])
        end
        local terra atomicAdd_nosync(sum : &float, value : float)
          @sum = @sum + value
        end
        b.atomicAdd_nosync = atomicAdd_nosync
        b.atomicAdd_sync = atomicAdd_sync
    else
        local terra atomicAdd_sync(sum : &double, value : double)
          C.pthread_mutex_lock(&[b.summutex_sym])
          @sum = @sum + value
          C.pthread_mutex_unlock(&[b.summutex_sym])
        end
        local terra atomicAdd_nosync(sum : &float, value : float)
          @sum = @sum + value
        end
        b.atomicAdd_nosync = atomicAdd_nosync
        b.atomicAdd_sync = atomicAdd_sync
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


local function makeGPULauncher(PlanData,kernelName,ft,compiledKernel, ispace) -- compiledKernel is the result of b.makeWrappedFunctions
-- TODO generalize to arbitrary number of threads
-- TODO current prevention of race-conditions in atomicAdd seems to be inefficient --> introduce separate sums for each thread
-- TODO make sure that arrays are traversed in column-major order
-- TODO make sure that granularity of thread-creation does not cause inefficiencies
    local numdims = #(ispace.dims)

    local struct thread_data {
      kmin : int[numdims],
      kmax : int[numdims],
      pd : &PlanData
    }

    local terra threadLauncher(threadarg : &opaque) : &opaque
        var threaddata = [&thread_data](threadarg)
        var pd = threaddata.pd
        var kmin = threaddata.kmin
        var kmax = threaddata.kmax

        compiledKernel(@pd, kmin, kmax)
    end
-- 
    local terra GPULauncher(pd : &PlanData)
        -- var [b.summutex_sym]
        C.pthread_mutex_init(&[b.summutex_sym], nil)


        var tdata1 : thread_data
        var tdata2 : thread_data

        var t1 : C.pthread_t
        var t2 : C.pthread_t

        -- TODO this is not the correct way to split up the work, will only work for one dimension.
        escape
            -- outermost dimension is split among threads
            local dimsize = ispace.dims[1].size
            emit quote
              tdata1.kmin[ 0 ] = 0
              tdata1.kmax[ 0 ] = dimsize/2

              tdata2.kmin[ 0 ] = dimsize/2
              tdata2.kmax[ 0 ] = dimsize
            end

          -- all other dimensions traverse everything
          for d = 2,numdims do
            local dimsize = ispace.dims[d].size
            emit quote
              tdata1.kmin[ [d-1] ] = 0
              tdata1.kmax[ [d-1] ] = dimsize

              tdata2.kmin[ [d-1] ] = 0
              tdata2.kmax[ [d-1] ] = dimsize
            end
          end
        end
        tdata1.pd = pd
        tdata2.pd = pd

        C.pthread_create(&t1, nil, threadLauncher, &tdata1)
        C.pthread_create(&t2, nil, threadLauncher, &tdata2)

        C.pthread_join(t1, nil)
        C.pthread_join(t2, nil)


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

      local launchquote = quote 
        kernel([pd_sym], [dimargs])
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

      local wrappedfunc = terra([pd_sym], [kminsym], [kmaxsym])
        [wrappedquote]
      end
      print(wrappedfunc)
      -- error()

      return wrappedfunc
    end

    print('\nInside util.makeGPUFunctions: The problemSpec.functions:')
    for k,v in pairs(problemSpec.functions) do print(k,v) end --debug 
    print('\nInside util.makeGPUFunctions: The problemSpec.functions[1]:')
    for k,v in pairs(problemSpec.functions[1]) do print(k,v) end --debug 
    print('\nInside util.makeGPUFunctions: The problemSpec.functions[1].typ:')
    for k,v in pairs(problemSpec.functions[1].typ) do print(k,v) end --debug 
    print('\nInside util.makeGPUFunctions: The problemSpec.functions[2]:')
    for k,v in pairs(problemSpec.functions[2]) do print(k,v) end --debug 
    print('\nInside util.makeGPUFunctions: The problemSpec.functions[2].typ:')
    for k,v in pairs(problemSpec.functions[2].typ) do print(k,v) end --debug 
    print('\nInside util.makeGPUFunctions: The problemSpec.functions[2].typ.__index:')
    for k,v in pairs(problemSpec.functions[2].typ.__index) do print(k,v) end --debug 
    print('\nInside util.makeGPUFunctions: The problemSpec:')
    for k,v in pairs(problemSpec.functions[1].typ.ispace.dims) do print(k,v) end
    for k,v in pairs(problemSpec.functions[1].typ.ispace.dims[1]) do print(k,v) end

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
