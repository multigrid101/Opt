local b = {}
local c = require('config')
local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#ifdef _WIN32
	#include <io.h>
#endif
]]

b.name = 'CPU'
b.numthreads = 1 -- DO NOT CHANGE THIS

b.threadarg = {}
b.threadarg_val = 1

-- atomicAdd START
if c.opt_float == float then
    local terra atomicAdd(sum : &float, value : float)
      @sum = @sum + value
    end
    b.atomicAdd_sync = atomicAdd
    b.atomicAdd_nosync = atomicAdd
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
        local terra atomicAdd(sum : &double, value : double)
          @sum = @sum + value
        end
        b.atomicAdd_sync = atomicAdd
        b.atomicAdd_nosync = atomicAdd
    else
        local terra atomicAdd(sum : &double, value : double)
          @sum = @sum + value
        end
        b.atomicAdd_sync = atomicAdd
        b.atomicAdd_nosync = atomicAdd
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
    -- kernelName = kernelName.."_"..tostring(ft)
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
      compiledKernel(@pd)
    end
    -- print(GPULauncher)
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
        wrappedquote = quote

          for [dimargs[k]] = 0, [dimsizes[k]] do
            [wrappedquote]
          end

        end
      end
      print(wrappedquote)

      local wrappedfunc = terra([pd_sym])
        [wrappedquote]
      end
      print(wrappedfunc)

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
