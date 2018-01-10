local libraryname, sourcedirectory, main, headerfile, outputname, embedsource = ...
embedsource = "true" == embedsource or false

local ffi = require("ffi")
local I = require(sourcedirectory .. '/ittnotify')

terralib.includepath = terralib.terrahome.."/include;."
local C,CN = terralib.includecstring[[ 
    #define _GNU_SOURCE
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #ifndef _WIN32
    #include <dlfcn.h>
    #include <libgen.h>
    #include <signal.h>
    sig_t SIG_DFL_fn() { return SIG_DFL; }
    #else
    #define NOMINMAX
    #include <windows.h>
    #include <Shlwapi.h>
    #endif
    #include "terra/terra.h"
]]

local LUA_GLOBALSINDEX = -10002

local tabsolutepath,setupsigsegv

if ffi.os == "Windows" then
    terra setupsigsegv(L : &C.lua_State) end
    terra tabsolutepath(rel : rawstring)
        var buf : rawstring = rawstring(C.malloc(C.MAX_PATH))
        C.GetFullPathNameA(rel,C.MAX_PATH,buf,nil)
        return buf
    end  
else

    local sigactionwrapper = ffi.os == "Linux" and "__sigaction_handler" or "__sigaction_u"
    local sigactionstruct = ffi.os == "Linux" and "sa_sigaction" or "__sa_sigaction"
    local terratraceback = global(&opaque -> {})
    
    local terra sigsegv(sig : int, info : &C.siginfo_t, uap : &opaque)
        C.signal(sig,C.SIG_DFL_fn())  --reset signal to default, just in case traceback itself crashes
        terratraceback(uap)
        C.raise(sig)
    end
    terra setupsigsegv(L : &C.lua_State)
    -- does not modify stack

        -- get terralib.traceback() and put on top of stack -- terralib | terralib.traceback()
        C.lua_getfield(L, LUA_GLOBALSINDEX,"terralib");
        C.lua_getfield(L, -1, "traceback");

        -- get copy of  terralib.traceback() from top of stack and cast to terra function -- terralib | terralib.traceback()
        var tb = C.lua_topointer(L,-1);
        if tb == nil then return end
        terratraceback = @[&(&opaque -> {})](tb)

        -- TODO what does this do?
        var sa : CN.sigaction
        sa.sa_flags = [terralib.constant(uint32,C.SA_RESETHAND)] or C.SA_SIGINFO
        C.sigemptyset(&sa.sa_mask)
        sa.[sigactionwrapper].[sigactionstruct] = sigsegv
        C.sigaction(C.SIGSEGV, &sa, nil)
        C.sigaction(C.SIGILL, &sa, nil)

        -- empty the stack
        C.lua_settop(L,-3)
    end
    
    terra tabsolutepath(path : rawstring)
        return C.realpath(path,nil)
    end
end
absolutepath = function(p) return ffi.string(tabsolutepath(p)) end

local Header = terralib.includec(headerfile)
local statename = libraryname.."_State"
local LibraryState = Header[statename]
assert(terralib.types.istype(LibraryState) and LibraryState:isstruct())

local apifunctions = terralib.newlist()
local apimatch = ("%s_(.*)"):format(libraryname)
for k,v in pairs(Header) do
    local name = k:match(apimatch)
    if name and terralib.isfunction(v) and name ~= "NewState" then
        local type = v:gettype()
        apifunctions[name] = {unpack(type.parameters,2)} -> type.returntype
    end
end

local wrappers = {}

struct LibraryState { L : &C.lua_State }

-- IMPORTANT: Must match Opt.h
struct Opt_InitializationParameters {
-- this struct collects all NON-PROBLEM-SPECIFIC information that is required by o.t at compile-time,
-- i.e. at the time when o.t is parsed (and all terra-functions within it are instantiated

    -- If true, all intermediate values and unknowns, are double-precision
    -- On platforms without double-precision float atomics, this 
    -- can be a drastic drag of performance.
    doublePrecision : int

    -- Valid Values: 0, no verbosity; 1, full verbosity
    verbosityLevel : int

    -- If true, a cuda timer is used to collect per-kernel timing information
    -- while the solver is running. This adds a small amount of overhead to every kernel.
    collectPerKernelTimingInfo : int

    -- possible values: 'backend_cpu', 'backend_gpu', backend_cpu_mt'
    backend : int8[20];

    -- only has effect for backend_cuda_mt, is set to 1 otherwise
    numthreads : int

    -- The opt-solver requires Multiplication of a vector with a linear operator
    -- "JTJ". We can either perform this multiplication with matrix-free code
    -- (default, if below option is false) or we can explicitly assemble the
    -- matrix for the linear operator (below option is true). (...cont. below)
    useMaterializedJTJ : int

    -- (... cont. from above). The multiplication (JTJ*p) can either be performed
    -- as (JT*(J*p)) (default) or (JT*J)*p. Set useFusedJTJ=true to use the latter
    -- variant. This will introduce a once-per-newton-solve to compute the nnz
    -- pattern of JT*J, and once-per-newton-step overhead to compute the values
    -- of JT*J. In return, the multiplication JTJ*p will (often) be cheaper than
    -- calculating (JT*(J*p)).
    -- This option has no effect if useMaterializedJTJ==false.
    useFusedJTJ : int
}

for name,type in pairs(apifunctions) do
    LibraryState.entries:insert { name, type }
end

local terra doerror(L : &C.lua_State)
    C.printf("%s\n",C.luaL_checklstring(L,-1,nil))
    C.lua_getfield(L,LUA_GLOBALSINDEX,"os")
    C.lua_getfield(L,-1,"exit")
    C.lua_pushnumber(L,1)
    C.lua_call(L,1,0)
    return nil
end

local sourcepath = absolutepath(sourcedirectory).."/?.t"
local terra NewState(params : Opt_InitializationParameters) : &LibraryState
-- loads o.t andcreates a new 'state' variable S. S is populated with (among other) freshly
-- terra-compiled versions of all api terra-functions. S is later passed to all
-- api C-functions, which in turns call the corresponding api terra-function in S.

-- example: OptProblem* Opt_ProblemDefine(OptState* S, params) {
--            S.ProblemDefine(params);
--          }

-- so this function can be summarized as follows:
-- terra NewState(params)
--   <do some stuff with the parameters>
  
--   opt = require('o.t')

--   S.ProblemDefine = opt.ProblemDefine
--   ...

--   return S
-- end


    -- We need this here to force the linker to put itt stuff into the executable.
    var name = I.__itt_string_handle_create("Opt_NewState()")
    var domain = I.__itt_domain_create("Main.Domain")
    I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name)
    I.__itt_task_end(domain)


    var S = [&LibraryState](C.malloc(sizeof(LibraryState)))
    var L = C.luaL_newstate();
    S.L = L
    if L == nil then return doerror(L) end
    C.luaL_openlibs(L)
    var o  = C.terra_Options { verbose = 0, debug = 1, usemcjit = 1 }
    
    if C.terra_initwithoptions(L,&o) ~= 0 then
        doerror(L)
    end
    setupsigsegv(L)

    -- Set global variables from Opt_InitializationParameters
    C.lua_pushboolean(L,params.doublePrecision);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_double_precision")

    var verbosityLevel : C.lua_Number = params.verbosityLevel
    C.lua_pushnumber(L,verbosityLevel);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_verbosity")

    C.lua_pushboolean(L,params.collectPerKernelTimingInfo);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_collect_kernel_timing")

    C.lua_pushstring(L,&(params.backend[0]))
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_backend")

    C.lua_pushnumber(L,params.numthreads);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_numthreads")

    C.lua_pushboolean(L,params.useMaterializedJTJ);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_use_materialized_jacobian")

    C.lua_pushboolean(L,params.useFusedJTJ);
    C.lua_setfield(L,LUA_GLOBALSINDEX,"_opt_use_fusedjtj")

    C.printf("NUMTHREADS=%d, jtj=%d\n", params.numthreads, params.useMaterializedJTJ)
    -- C.exit(0)
    -- stack is now empty

    -- push 'package()' onto stack -- package
    C.lua_getfield(L,LUA_GLOBALSINDEX,"package")

    escape 
        if embedsource then -- this branch is for libOpt.a
            emit quote C.lua_getfield(L,-1,"preload") end -- package | package.preload()
			
            local listdir_command = ""
            if ffi.os == "Windows" then
                    listdir_command = "cmd /c dir /b "
            else
                    listdir_command = "ls "
            end 

            -- TODO is this important or debug?
            print(listdir_command..sourcedirectory)
			
            -- TODO what exactly happens in this loop, the stackvalues do not seem to be used
            for fullfilename in io.popen(listdir_command..sourcedirectory):lines() do -- iterate over filenames in sourcedir

                -- open each valid terra file
                local filenamestub = fullfilename:match("(.*)%.t")
                if filenamestub then
                    local filecontent = io.open(sourcedirectory.."/"..fullfilename,"r"):read("*all")
                    emit quote
                        -- stack after following line: package | package.preload | compiledchunck
                        if 0 ~= C.terra_loadbuffer(L,filecontent,[#filecontent],["@"..fullfilename]) then doerror(L) end

                        -- stack after following line: package | compiledchuck
                        C.lua_setfield(L,-2,filenamestub)
                    end
                end
            end
        else -- this branch is for libOptDev.a
            emit quote 
                C.lua_getfield(L,-1,"terrapath")
                C.lua_pushstring(L,";")
                C.lua_pushstring(L,sourcepath)
                C.lua_concat(L,3)
                C.lua_setfield(L,-2,"terrapath")
            end
        end
    end
    
    -- top of stack after next two lines: -- ... | require() | 'o' (o.t)
    C.lua_getfield(L,LUA_GLOBALSINDEX,"require")
    C.lua_pushstring(L,main)

    -- calls 'require(o.t)' and stores result on top of stack
    -- result is a sollection of the api functions, i.e. a table 'opt' with
    -- opt.ProblemDefine()
    -- opt.ProblemPlan()
    -- etc. etc. (see o.t)
    if C.lua_pcall(L,1,1,0) ~= 0 then return doerror(L) end
    
    -- stores ctype pointers to all api functions from o.t in S
    escape -- top of stack: ... | opt
        for apifunc_name,apifunc_terratype in pairs(apifunctions) do
            emit quote
                C.lua_getfield(L,-1,apifunc_name) -- ... | opt | opt.ProblemDefine
                C.lua_getfield(L,-1,"getpointer") -- ... | opt | opt.ProblemDefine | opt.ProblemDefine.getpointer
                C.lua_insert(L,-2) -- ... | opt | opt.ProblemDefine.getpointer | opt.ProblemDefine
                C.lua_call(L,1,1) -- ... opt | result of opt.ProblemDefine.getpointer(opt.ProblemDefine)
                S.[apifunc_name] = @[&apifunc_terratype](C.lua_topointer(L,-1))
                C.lua_settop(L, -2) -- ...
            end
        end
    end
    return S
end
print(NewState)
wrappers[libraryname.."_NewState"] =  NewState

for k,type in pairs(apifunctions) do
    local syms = type.type.parameters:map(symbol)
    local terra wfn(state : &LibraryState, [syms]) : type.type.returntype
        return state.[k]([syms])
    end
    wrappers[libraryname.."_"..k] = wfn 
end 

local flags = {}
if ffi.os == "Windows" then
    flags = terralib.newlist { string.format("/IMPLIB:%s.lib",libraryname),terralib.terrahome.."\\lib\\terra.lib",terralib.terrahome.."\\lib\\lua51.lib","Shlwapi.lib", '-lpthread'}
    
    for k,_ in pairs(wrappers) do
        flags:insert("/EXPORT:"..k)
    end
end
terralib.saveobj(outputname,wrappers,flags)
