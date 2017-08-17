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

b.name = 'CUDA'
b.numthreads = 1 -- DO NOT CHANGE THIS

b.threadarg = {}

-- atomicAdd START
if c.opt_float == float then
    local terra atomicAdd(sum : &float, value : float)
    	terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, value)
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
            var address_as_i : uint64 = [uint64] (sum);
            terralib.asm(terralib.types.unit,"red.global.add.f64 [$0],$1;","l,d", true, address_as_i, value)
        end
        b.atomicAdd_sync = atomicAdd
        b.atomicAdd_nosync = atomicAdd
    else
        local terra atomicAdd(sum : &double, value : double)
            var address_as_i : &uint64 = [&uint64] (sum);
            var old : uint64 = address_as_i[0];
            var assumed : uint64;

            repeat
                assumed = old;
                old = terralib.asm(uint64,"atom.global.cas.b64 $0,[$1],$2,$3;", 
                    "=l,l,l,l", true, address_as_i, assumed, 
                    __double_as_ull( value + __ull_as_double(assumed) )
                    );
            until assumed == old;

            return __ull_as_double(old);
        end
        b.atomicAdd_sync = atomicAdd
        b.atomicAdd_nosync = atomicAdd
    end
end
-- atomicAdd END

-- Using the "Kepler Shuffle", see http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
-- TODO used also in solver, so find appropriate place
-- warpReduce() START
local __shfl_down 
local warpSize = 32
if opt_float == float then
    terra __shfl_down(v : float, delta : uint, width : int)
    	var ret : float;
        var c : int;
    	c = ((warpSize-width) << 8) or 0x1F;
    	ret = terralib.asm(float,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, v, delta, c)
    	return ret;
    end
else
    struct uint2 {
        x : uint32;
        y : uint32;
    }

    struct uint2Double {
        union {
            u2 : uint2;
            d: double;
        }
    }

    terra __shfl_down(v : double, delta : uint, width : int)
        var ret : uint2Double;
        var init : uint2Double;
        init.d = v
        var c : int;
        c = ((warpSize-width) << 8) or 0x1F;
        ret.u2.x = terralib.asm(uint32,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, init.u2.x, delta, c)
        ret.u2.y = terralib.asm(uint32,"shfl.down.b32 $0, $1, $2, $3;","=f,f,r,r", true, init.u2.y, delta, c)
        return ret.d;
    end
end
local terra warpReduce(val : opt_float) 

  var offset = warpSize >> 1
  while offset > 0 do 
    val = val + __shfl_down(val, offset, warpSize);
    offset =  offset >> 1
  end
-- Is unrolling worth it?
  return val;

end
b.warpReduce = warpReduce
-- warpReduce() END

-- landeid() is the 'local' id of a thread within a warp
terra b.laneid()
    var laneid : int;
    laneid = terralib.asm(int,"mov.u32 $0, %laneid;","=r", true)
    return laneid;
end

-- allocate, memset and memcpy START
b.allocateHost = macro(function(ptrptr, numbytes, datatype)
  return quote @ptrptr = C.malloc(numbytes) end
end)

b.allocateDevice = macro(function(ptrptr, numbytes, datatype)
  return `C.cudaMalloc([&&opaque](ptrptr), numbytes)
end)

b.memsetDevice = macro(function(ptr, value, numbytes)
  return `C.cudaMemset([&opaque](ptr), value, numbytes)
end)

b.memcpyDevice = macro(function(targetptr, sourceptr, numbytes)
  return `C.cudaMemcpy(targetptr, sourceptr, numbytes, C.cudaMemcpyDeviceToDevice)
end)

b.memcpyDevice2Host = macro(function(targetptr, sourceptr, numbytes)
  return `C.cudaMemcpy(targetptr, sourceptr, numbytes, C.cudaMemcpyDeviceToHost)
end)
-- allocate, memset and memcpy END

function b.makeIndexInitializer(Index, dims, dimnames, fieldnames)

    local initIndex
    if #dims <= 3 then
        local dimnames = "xyz"
        terra initIndex(self : &Index) : bool -- add 'x', 'y' and 'z' field to the index
            escape
                local lhs,rhs = terralib.newlist(),terralib.newlist()
                local valid = `true
                for i = 1,#dims do
                    local name = dimnames:sub(i,i)
                    local l = `self.[fieldnames[i]]
                    local r = `blockDim.[name] * blockIdx.[name] + threadIdx.[name]
                    lhs:insert(l)
                    rhs:insert(r)
                    valid = `valid and l < [dims[i].size]
                end
                emit quote
                    [lhs] = [rhs]
                    return valid
                end
            end  
        end
        -- print(Index) -- debug
        -- for k,v in pairs(Index.methods)  do print(k,v) end -- debug
    end

    return initIndex

end

function b.getKernelArglist(indexspace) -- TODO should this result be cached or do we need new symbols all the time?
    -- local numdims = indexspace:getDimensionality()
    -- local arglist = terralib.newlist()
    -- for k = 0, numdims do
    --     arglist:insert(symbol(int, 'x' .. tostring(k)))
    -- end
    local arglist = terralib.newlist()
    return arglist
end

------------------------------------------------------------------------------ wrap and compile kernels
-- TODO choose better name and put with other general purpose stuff
-- TODO this seems to be re-defined in solver AND o.t
local cd = macro(function(apicall) 
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var str = [apicallstr]
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            C.printf("In call: %s", str)
            C.printf("In file: %s\n", filename)
            C.exit(r)
        end
    in
        r
    end end)
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
    local kernelparams = compiledKernel:gettype().parameters
    local params = terralib.newlist {}
    for i = 3,#kernelparams do --skip GPU launcher and PlanData
        params:insert(symbol(kernelparams[i]))
    end
    local function createLaunchParameters(pd)
        if ft.kind == "CenteredFunction" then
            local ispace = ft.ispace
            local exps = terralib.newlist()
            for i = 1,3 do
               local dim = #ispace.dims >= i and ispace.dims[i].size or 1
                local bs = GRID_SIZES[#ispace.dims][i]
                exps:insert(dim)
                exps:insert(bs)
            end
            return exps
        else
            return {`pd.parameters.[ft.graphname].N,256,1,1,1,1}
        end
    end
    local terra GPULauncher(pd : &PlanData, [params])
        var xdim,xblock,ydim,yblock,zdim,zblock = [ createLaunchParameters(pd) ]
            
        var launch = terralib.CUDAParams { (xdim - 1) / xblock + 1, (ydim - 1) / yblock + 1, (zdim - 1) / zblock + 1, 
                                            xblock, yblock, zblock, 
                                            0, nil }
        var stream : C.cudaStream_t = nil
        var endEvent : C.cudaEvent_t 
        if ([_opt_collect_kernel_timing]) then
            pd.timer:startEvent(kernelName,nil,&endEvent)
        end

        checkedLaunch(kernelName, compiledKernel(&launch, @pd, params))
        
        if ([_opt_collect_kernel_timing]) then
            pd.timer:endEvent(nil,endEvent)
        end

        cd(C.cudaGetLastError())
    end
    return GPULauncher
end

function b.makeWrappedFunctions(problemSpec, PlanData, delegate, names) -- same  problemSpec as in solver.t

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
    
    for _,problemfunction in ipairs(problemSpec.functions) do -- problemfunction is of type ProblemFunction, see grammar in o.t
        if problemfunction.typ.kind == "CenteredFunction" then
           local ispace = problemfunction.typ.ispace
           local dimcount = #ispace.dims
	       assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
           local ks = delegate.CenterFunctions(ispace,problemfunction.functionmap) -- ks are the kernelfunctions as shown in gaussNewtonGPU.t
           for name,func in pairs(ks) do
             -- print(name,func) -- debug
                kernelFunctions[getkname(name,problemfunction.typ)] = { kernel = func , annotations = { {"maxntidx", GRID_SIZES[dimcount][1]}, {"maxntidy", GRID_SIZES[dimcount][2]}, {"maxntidz", GRID_SIZES[dimcount][3]}, {"minctasm",1} } }
           end
        else
            local graphname = problemfunction.typ.graphname
            local ispace = problemfunction.typ.ispace -- by SO
            -- local ks = delegate.GraphFunctions(graphname,problemfunction.functionmap) -- original
            local ks = delegate.GraphFunctions(graphname, problemfunction.functionmap,nil, ispace) -- by SO
            for name,func in pairs(ks) do            
                kernelFunctions[getkname(name,problemfunction.typ)] = { kernel = func , annotations = { {"maxntidx", 256}, {"minctasm",1} } }
            end
        end
    end
    
    -- print('\nIn makeGPUFunctions:') -- debug
    -- for k,v in pairs(kernelFunctions) do print(k,v) end
    local kernels = terralib.cudacompile(kernelFunctions, false)
    -- print('\ncompiled cuda kernels')
    -- for k,v in pairs(kernels) do print(k,v) end -- end
    
    -- step 2: generate wrapper functions around each named thing
    local grouplaunchers = {}
    for _,name in ipairs(names) do -- name = e.g. 'PCGInit1', 'PCGStep1', etc.
        local args
        local launches = terralib.newlist()
        for _,problemfunction in ipairs(problemSpec.functions) do
            local kname = getkname(name,problemfunction.typ)
            print(kname)
            local kernel = kernels[kname]
            if kernel then -- some domains do not have an associated kernel, (see _Finish kernels in GN which are only defined for 
                local launcher = makeGPULauncher(PlanData, name, problemfunction.typ, kernel)
                -- print(launcher) -- debug
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
            fn = terra([args]) launches end
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
