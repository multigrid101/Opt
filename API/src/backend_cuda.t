local b = {}
local S = require("std")
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

    CUsp = terralib.includecstring [[
        #include <cusparse.h>
    ]]

b.name = 'CUDA'
b.numthreads = 1 -- DO NOT CHANGE THIS

b.threadarg = {}
b.threadarg_val = 1

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
local struct Event {
	startEvent : C.cudaEvent_t
	endEvent : C.cudaEvent_t
	duration : float
	eventName : rawstring
}
b.Event = Event

local TimerEvent = C.cudaEvent_t

local struct Timer {
	eventList : &Array(Event)
}
b.Timer = Timer
-- for k,v in pairs(C) do print(k,v) end
-- print(C.cudaEvent_t)
-- print('ASDF')
-- print(C.CUevent_st)

-- print(Timer)
-- -- C.cudaEvent_t:printpretty()
-- error()

terra Timer:init() 
	self.eventList = [Array(Event)].alloc():init()
end

terra Timer:cleanup()
    for i = 0,self.eventList:size() do
        var eventInfo = self.eventList(i);
        C.cudaEventDestroy(eventInfo.startEvent);
        C.cudaEventDestroy(eventInfo.endEvent);
    end
	self.eventList:delete()
end 


-- terra Timer:startEvent(name : rawstring,  stream : C.cudaStream_t, endEvent : &C.cudaEvent_t)
terra Timer:startEvent(name : rawstring,  eventptr : &Event)
    (@eventptr).eventName = name

    C.cudaEventCreate(&(@eventptr).startEvent)
    C.cudaEventCreate(&(@eventptr).endEvent)

    C.cudaEventRecord((@eventptr).startEvent, nil)
    -- @endEvent = timingInfo.endEvent


    -- var starttime : double
    -- starttime = start.tv_sec
    -- starttime = starttime + [double](start.tv_nsec)



    -- TODO clean this up a little
    -- C.printf('starting kernel %s at time %lu\n', name, starttime)
end
terra Timer:endEvent(eventptr : &Event, dummy : int)
-- dummy arg is required for mt backend
    C.cudaEventRecord((@eventptr).endEvent, nil)
    self.eventList:insert(@eventptr)


    -- C.printf('    stopping kernel at time %lu\n', stoptime)
end

-- TODO only used in next function, so make local there
terra isprefix(pre : rawstring, str : rawstring) : bool
    if @pre == 0 then return true end
    if @str ~= @pre then return false end
    return isprefix(pre+1,str+1)
end
terra Timer:evaluate()
            -- C.printf("asdfasdf\n")
	if ([c._opt_verbosity > 0]) then
          var aggregateTimingInfo = [Array(tuple(float,int))].salloc():init()
          var aggregateTimingNames = [Array(rawstring)].salloc():init()

            C.printf("asdf %d\n", self.eventList:size())
          for i = 0,self.eventList:size() do
            var eventInfo = self.eventList(i);
            C.cudaEventSynchronize(eventInfo.endEvent)
            C.cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
            var index =  aggregateTimingNames:indexof(eventInfo.eventName)
            if index < 0 then
              aggregateTimingNames:insert(eventInfo.eventName)
              aggregateTimingInfo:insert({eventInfo.duration, 1})
            else
              aggregateTimingInfo(index)._0 = aggregateTimingInfo(index)._0 + eventInfo.duration
              aggregateTimingInfo(index)._1 = aggregateTimingInfo(index)._1 + 1
            end
          end

          C.printf(		"--------------------------------------------------------\n")
          C.printf(		"        Kernel        |   Count  |   Total   | Average \n")
          C.printf(		"----------------------+----------+-----------+----------\n")
          for i = 0, aggregateTimingNames:size() do
              C.printf(	"----------------------+----------+-----------+----------\n")
              C.printf(" %-20s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames(i), aggregateTimingInfo(i)._1, aggregateTimingInfo(i)._0, aggregateTimingInfo(i)._0/aggregateTimingInfo(i)._1)
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
print(Timer.methods.evaluate)
-- error()
--------------------------- Timing stuff end

---------------- ReduceVar start
b.ReduceVar = &opt_float
b.ReduceVarHost = &opt_float

b.ReduceVarHost.allocate2 = function(variable) -- modified method name to avoid name clash with ReduceVar methods
    return quote [variable] = [&opt_float](C.malloc(sizeof(opt_float))) end
end

b.ReduceVarHost.getData2 = function( varquote, index ) -- modified method name to avoid name clash with ReduceVar methods
  return `@[varquote]
end

b.ReduceVarHost.reduceAllThreads2 = function( varquote ) -- modified method name to avoid name clash with ReduceVar methods
  return {}
end
---------------------------------------------

b.ReduceVar.reduceAllThreads = function( varquote )
  return {}
end


b.ReduceVar.allocate = function(variable)
  -- if location == 0 then -- Device
    return `C.cudaMalloc([&&opaque](&([variable])), sizeof(opt_float))
  -- else
    -- return quote [variable] = C.malloc([&&opaque](variable), sizeof(opt_float))
  -- end
end

b.ReduceVar.getData = function(variablequote, k)
  return `(@([variablequote]))
end

b.ReduceVar.getDataPtr = function(variable, k)
  -- C.printf('starting getdataptr\n')
  -- C.printf('stopping getdataptr\n')
  return variable
end


b.ReduceVar.setToConst = function(variable, val)
  return `C.cudaMemset([&opaque]( [variable] ), [val], sizeof(opt_float))
end

b.ReduceVar.memcpyDevice2Host = function(targetquote, sourcequote)
  return `C.cudaMemcpy( [targetquote] , [sourcequote] , sizeof(opt_float), C.cudaMemcpyDeviceToHost)
end

b.ReduceVar.reduceAllThreads = function( varquote )
  return {}
end

b.ReduceVar.memcpyDevice = function(targetquote, sourcequote)
  return `C.cudaMemcpy([&opaque]([targetquote]), [&opaque]([sourcequote]), sizeof(opt_float), C.cudaMemcpyDeviceToDevice)
end
---------------- ReduceVar end

-- atomicAdd START
if c.opt_float == float then
    local terra atomicAddSync(sum : &float, value : float, offset : int)
    	terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, value)
    end
    local terra atomicAddNosync(sum : &float, value : float)
    	terralib.asm(terralib.types.unit,"red.global.add.f32 [$0],$1;","l,f", true, sum, value)
    end
    b.atomicAdd_sync = atomicAddSync
    b.atomicAdd_nosync = atomicAddNosync
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
        local terra atomicAddSync(sum : &double, value : double, offset : int)
            var address_as_i : uint64 = [uint64] (sum);
            terralib.asm(terralib.types.unit,"red.global.add.f64 [$0],$1;","l,d", true, address_as_i, value)
        end
        local terra atomicAddNosync(sum : &double, value : double)
            var address_as_i : uint64 = [uint64] (sum);
            terralib.asm(terralib.types.unit,"red.global.add.f64 [$0],$1;","l,d", true, address_as_i, value)
        end
        b.atomicAdd_sync = atomicAddSync
        b.atomicAdd_nosync = atomicAddNosync
    else
        local terra atomicAddSync(sum : &double, value : double, offset : int)
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
        local terra atomicAddNoSync(sum : &double, value : double)
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
        b.atomicAdd_sync = atomicAddSync
        b.atomicAdd_nosync = atomicAddNoSync
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
-- TODO should we maybe put the cd() stuff around cuda calls in here? that way,
-- error checking would be hidden in the backend.
-- Discussion:
--    * leaving error checking in solverGPU... gets us more precise location of the error
--    * putting error checking here would hide it from the user/reader and ensure
--      that it is always done

-- datatype arg is not used here but required for other backends
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

b.memcpyHost2Device = macro(function(targetptr, sourceptr, numbytes)
  return `C.cudaMemcpy(targetptr, sourceptr, numbytes, C.cudaMemcpyHostToDevice)
end)
-- allocate, memset and memcpy END

-- TODO doesn't work with double precision
-- cusparse stuff START -- TODO change name
local function insertMatrixlibEntries(PlanData_t)
-- this function will simply insert &opaque of value nil for other backends
  PlanData_t.entries:insert {"handle", CUsp.cusparseHandle_t }
  PlanData_t.entries:insert {"desc", CUsp.cusparseMatDescr_t }
end
b.insertMatrixlibEntries = insertMatrixlibEntries


local terra computeNnzPatternATA(handle : CUsp.cusparseHandle_t, -- needed by cusparse lib TODO refactor
                                descr : CUsp.cusparseMatDescr_t, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                rowPtrA : &int, colIndA : &int,
                                rowPtrATA : &int, ptrTo_colIndATA : &&int, nnzATAptr : &int) 
-- computes only rowPtrATA and allocates colIndATA, but doesn't actually compute it.
-- colIndATA is computed by computeATA()
                      
  cd(CUsp.cusparseXcsrgemmNnz(handle, CUsp.CUSPARSE_OPERATION_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                        nUnknowns, nUnknowns, nResiduals,
                        descr, nnzA, rowPtrA, colIndA,
                        descr, nnzA, rowPtrA, colIndA,
                        descr, rowPtrATA, nnzATAptr))

  cd( b.allocateDevice(ptrTo_colIndATA, @nnzATAptr*sizeof(int), int) )
end
b.computeNnzPatternATA = computeNnzPatternATA


local terra computeATA(handle : CUsp.cusparseHandle_t, -- needed by cusparse lib TODO refactor
                                descr : CUsp.cusparseMatDescr_t, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                valA : &float, rowPtrA : &int, colIndA : &int,
                                valATA : &float, rowPtrATA : &int, colIndATA : &int) -- valATA(out), rowATA(int), colATA(out)

  cd(CUsp.cusparseScsrgemm(handle, CUsp.CUSPARSE_OPERATION_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
           nUnknowns,nUnknowns,nResiduals,
           descr, nnzA, valA, rowPtrA, colIndA,
           descr, nnzA, valA, rowPtrA, colIndA,
           descr, valATA, rowPtrATA, colIndATA ))
end
b.computeATA = computeATA


local terra computeAT(handle : CUsp.cusparseHandle_t, -- needed by cusparse lib TODO refactor
                                descr : CUsp.cusparseMatDescr_t, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                valA : &float, rowPtrA : &int, colIndA : &int,
                                valAT : &float, rowPtrAT : &int, colIndAT : &int) -- valATA(out), rowATA(int), colATA(out)

            cd(CUsp.cusparseScsr2csc(handle, nResiduals, nUnknowns, nnzA,
                                     valA, rowPtrA, colIndA,
                                     valAT, colIndAT, rowPtrAT,
                                     CUsp.CUSPARSE_ACTION_NUMERIC,CUsp.CUSPARSE_INDEX_BASE_ZERO))
end
b.computeAT = computeAT


local terra applyAtoVector(handle : CUsp.cusparseHandle_t, -- needed by cusparse lib TODO refactor
                                descr : CUsp.cusparseMatDescr_t, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                valA : &float, rowPtrA : &int, colIndA : &int,
                                valInVec : &float, valOutVec : &float) -- valInVec(in), valOutVec(out)

  var consts = array(0.f,1.f,2.f)
  cd(CUsp.cusparseScsrmv(
              handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
              nResiduals, nUnknowns, nnzA,
              &consts[1], descr,
              valA, rowPtrA, colIndA,
              [&float](valInVec),
              &consts[0], [&float](valOutVec)
          ))
end
b.applyAtoVector = applyAtoVector


local terra initMatrixStuff(handlePtr : &CUsp.cusparseHandle_t, descrPtr : &CUsp.cusparseMatDescr_t)
-- do some init stuff for the cusparse library, this func will most likely just get nil pointers in other
-- backends and do nothing.
  cd(CUsp.cusparseCreateMatDescr( descrPtr ))
  cd(CUsp.cusparseSetMatType( @descrPtr,CUsp.CUSPARSE_MATRIX_TYPE_GENERAL ))
  cd(CUsp.cusparseSetMatIndexBase( @descrPtr,CUsp.CUSPARSE_INDEX_BASE_ZERO ))
  cd(CUsp.cusparseCreate( handlePtr ))
end
b.initMatrixStuff = initMatrixStuff
-- cusparse stuff END


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


function b.make_Image_initGPU(imagetype_terra)
    local initGPU= terra(self : &imagetype_terra)
        var data : &uint8 -- we cast this to the correct type later inside setGPUptr

        b.cd( b.allocateDevice(&data, self:totalbytes(), uint8) )
        b.cd( b.memsetDevice(data, 0, self:totalbytes()) )

        self:initFromGPUptr(data) -- (short explanataion): set self.data = data (and cast to appropriate ptr-type)
    end
    initGPU:setname('Image.initGPU')

   return initGPU
end

-- IMAGE SPECIALIZATION START
function b.make_Image_metamethods__apply(imagetype_terra, indextype_terra, vectortype_terra, loadAsVector, VT)
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
    local metamethods__update
    if loadAsVector then
        metamethods__update = terra(self : &imagetype_terra, idx : indextype_terra, v : vectortype_terra)
      -- TODO backend-specific
            VT(self.data)[idx:tooffset()] = @VT(&v)
        end
    else
        metamethods__update = terra(self : &imagetype_terra, idx : indextype_terra, v : vectortype_terra)
      -- TODO backend-specific
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
        var endEvent : Event
        if ([_opt_collect_kernel_timing]) then
            pd.timer:startEvent(kernelName,&endEvent)
        end

        checkedLaunch(kernelName, compiledKernel(&launch, @pd, params))
        
        if ([_opt_collect_kernel_timing]) then
            pd.timer:endEvent(&endEvent, 0)
        end

        -- C.printf('bla1\n')
        cd(C.cudaGetLastError())
        -- C.printf('bla2\n')
    end
    print(checkedLaunch)
    -- if kernelName == 'computeCost_W_H' then print(GPULauncher) error() end
    return GPULauncher
end

function b.makeWrappedFunctions(problemSpec, PlanData, delegate, names) -- same  problemSpec as in solver.t

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
    
    for _,problemfunction in ipairs(problemSpec.functions) do -- problemfunction is of type ProblemFunction, see grammar in o.t
        if problemfunction.typ.kind == "CenteredFunction" then
           local ispace = problemfunction.typ.ispace
           local dimcount = #ispace.dims
	       assert(dimcount <= 3, "cannot launch over images with more than 3 dims")
           local ks = delegate.CenterFunctions(ispace,problemfunction.functionmap) -- ks are the kernelfunctions as shown in gaussNewtonGPU.t
           for name,func in pairs(ks) do
             print(name,func) -- debug
                kernelFunctions[getkname(name,problemfunction.typ)] = { kernel = func , annotations = { {"maxntidx", GRID_SIZES[dimcount][1]}, {"maxntidy", GRID_SIZES[dimcount][2]}, {"maxntidz", GRID_SIZES[dimcount][3]}, {"minctasm",1} } }
           end
           -- error()
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
            fn = terra([args])
              -- C.printf('starting launcher\n')
              launches
              -- C.printf('stopping launcher\n')
            end

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
    -- error()

    return grouplaunchers
end
-------------------------------- END wrap and compile kernels

return b
