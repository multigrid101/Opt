local ad = require('ad')
local A = ad.classes
local optlib = require("lib")
local util = require('util')
local c = require('config')

TRACE = true -- TODO added by SO to turn on/off lots of print-statements required to understand how the code works
function printt(thing)
  print('the name of the thing:')
  if TRACE then
    print(thing)
  end
  print('the thing as table:')
  for k,v in pairs(thing) do print(k,v) end
end

opt = {}


local List = terralib.newlist


local PROBLEM_STAGES  = { inputs = 0, functions = 1 }

if c.verboseAD then
	logAD = macro(function(fmt,...)
		local args = {...}
		return `C.printf(fmt, args)
	end)
	dprint = print
else
	logAD = macro(function(fmt,...)
		return 0
	end)
	dprint = function() end
end



A:Extern("ExpLike",function(x) return ad.Exp:isclassof(x) or ad.ExpVector:isclassof(x) end)
A:Define [[
Dim = (string name, number size, number? _index) unique
IndexSpace = (Dim* dims) unique
Index = Offset(number* data) unique
      | GraphElement(any graph, string element) unique
ImageType = (IndexSpace ispace, TerraType scalartype, number channelcount) unique
ImageLocation = ArgumentLocation(number idx) | UnknownLocation | StateLocation
Image = (string name, ImageType type, boolean scalar, ImageLocation location)
ProblemParam = ImageParam(ImageType imagetype, boolean isunknown)
             | ScalarParam(TerraType type)
             | GraphParam(TerraType type)
             attributes (string name, any idx)

VarDef =  ImageAccess(Image image,  Shape _shape, Index index, number channel) unique
       | BoundsAccess(Offset min, Offset max) unique
       | IndexValue(number dim, number shift_) unique
       | ParamValue(string name,TerraType type) unique

FunctionKind = CenteredFunction(IndexSpace ispace) unique
             | GraphFunction(string graphname) unique

ResidualTemplate = (Exp expression, ImageAccess* unknowns)
EnergySpec = (FunctionKind kind, ResidualTemplate* residuals)

FunctionSpec = (FunctionKind kind, string name, string* arguments, ExpLike* results, Scatter* scatters, EnergySpec? derivedfrom)

Scatter = (Image image,Index index, number channel, Exp expression, string kind)

ProblemSpec = ()
ProblemSpecAD = ()

UnknownType = (ImageParam* images)
]]
Dim = A.Dim
IndexSpace = A.IndexSpace
Index = A.Index
Offset = A.Offset
GraphElement = A.GraphElement
ImageType = A.ImageType
ImageLocation = A.ImageLocation
Image = A.Image
ProblemParam = A.ImageParam
ImageParam = A.ImageParam
VarDef = A.VarDef
ImageAccess = A.ImageAccess
BoundsAccess = A.BoundsAccess
IndexValue = A.IndexValue
ParamValue = A.ParamValue
FunctionKind = A.FunctionKind
CenterFunction = A.CenterFunction
GraphFunction = A.GraphFunction
ResidualTemplate = A.ResidualTemplate
EnergySpec = A.EnergySpec
FunctionSpec = A.FunctionSpec
Scatter = A.Scatter
ProblemSpec = A.ProblemSpec
ProblemSpecAD = A.ProblemSpecAD
UnknownType = A.UnknownType


-- TODO find an appropriate place for this
local function MapAndGroupBy(list,fn,...)
    local groups,map = List(),{}
    for _,l in ipairs(list) do
        local g,v = fn(l,...)
        if not map[g] then
            map[g] = List()
            groups:insert(g)
        end
        map[g]:insert(v)
    end
    return groups,map
end

--------------------------- START UnknownType ----------------------------
function UnknownType:init()
    self.ispaces,self.ispacetoimages = MapAndGroupBy(self.images, function(ip)
        assert(ip.imagetype.scalartype == opt_float, "unknowns must be floating point numbers")
        return ip.imagetype.ispace, ip
    end)
    self.ispacesizes = {}
    for _,ispace in ipairs(self.ispaces) do
        local N = 0
        for _,ip in ipairs(self.ispacetoimages[ispace]) do
            N = N + ip.imagetype.channelcount
        end
        self.ispacesizes[ispace] = N
    end
end

function UnknownType:VectorSizeForIndexSpace(ispace) return assert(self.ispacesizes[ispace],"unused ispace") end 

function UnknownType:UnknownIteratorForIndexSpace(ispace)
    local images = self.ispacetoimages[ispace]
    local i,j,c = 0,1,0
    return function()
        if c >= images[j].imagetype.channelcount then
            j,c = j+1,0
        end
        if j > #images then return nil end
        i,c = i + 1,c + 1
        --print(i-1,images[j].name,c-1)
        return i - 1, images[j].name,c - 1
    end
end
--------------------------- END UnknownType ----------------------------

--------------------------- START CenteredFunction ----------------------------
function A.CenteredFunction:__tostring() return tostring(self.ispace) end
--------------------------- END CenteredFunction ----------------------------

--------------------------- START GraphFunction ----------------------------
function A.GraphFunction:__tostring() return tostring(self.graphname) end
--------------------------- END GraphFunction ----------------------------

local function extractresidualterms(...)
    local exp = terralib.newlist {}
    for i = 1, select("#",...) do
        local e = select(i,...)
        if ad.ExpVector:isclassof(e) then
            for i,t in ipairs(e:expressions()) do
                t = assert(ad.toexp(t), "expected an ad expression")
                exp:insert(t)
            end
        else
            exp:insert((assert(ad.toexp(e), "expected an ad expression")))
        end
    end
    return exp
end

local function bboxforexpression(ispace,exp)
    local usesbounds = false
    local bmin,bmax = ispace:ZeroOffset(),ispace:ZeroOffset()
    exp:visit(function(a)
        if ImageAccess:isclassof(a) then
            assert(Offset:isclassof(a.index,"bbox not defined for graphs"))
            if a.image.gradientimages then
                local shiftedbbox = a.image.bbox:shift(a.index)
                bmin,bmax = bmin:Min(shiftedbbox.min),bmax:Max(shiftedbbox.max)
            else
                bmin,bmax = bmin:Min(a.index),bmax:Max(a.index)
            end
        elseif BoundsAccess:isclassof(a) then
            usesbounds = true
        end
    end)
    if usesbounds then 
        bmin,bmax = ispace:ZeroOffset(),ispace:ZeroOffset()
    end
    return BoundsAccess(bmin,bmax)
end

local function classifyexpression(exp) -- what index space, or graph is this thing mapped over
    local classification
    local seenunknown = {}
    local unknownaccesses = terralib.newlist()
    local function addunknown(u)
        if not seenunknown[u] then
            unknownaccesses:insert(u)
            seenunknown[u] = true
        end
    end
    exp:visit(function(a)
        if ImageAccess:isclassof(a) then -- assume image X is unknown
            if a.image.location == A.UnknownLocation then
                addunknown(a)
            elseif a.image.gradientimages then
                for i,im in ipairs(a.image.gradientimages) do
                    assert(Offset:isclassof(a.index),"NYI - precomputed with graphs")
                    addunknown(im.unknown:shift(a.index))
                end
            end
            local aclass = Offset:isclassof(a.index) and A.CenteredFunction(a.image.type.ispace) or A.GraphFunction(a.index.graph.name)
            assert(nil == classification or aclass == classification, "residual contains image reads from multiple domains")
            classification = aclass
        end
    end)
    local template = A.ResidualTemplate(exp,unknownaccesses)
    if not classification then
        error("residual must actually use some image")
    end
    if classification.kind == "CenteredFunction" then
        exp:visit(function(a)
            if BoundsAccess:isclassof(a) and #a.min.data ~= #classification.ispace.dims then
                error(string.format("%s does not match index space %s",a,classification.ispace.dims))
            end
        end)
        -- by default zero-out any residual computation that uses out-of-bounds things
        -- users can opt for per-residual custom behavior using the InBounds checks
        local bbox = bboxforexpression(classification.ispace,exp)
        template.expression = ad.select(bbox:asvar(),exp,0)
    end
    return classification,template
end

-- TODO Important function, move downwards
local function toenergyspecs(Rs)    
    local kinds,kind_to_templates = MapAndGroupBy(Rs,classifyexpression)
    return kinds:map(function(k) return A.EnergySpec(k,kind_to_templates[k]) end)
end

-------------------- More weird random stuff start
function opt.InBounds(...)
    local offset = Offset(List{...})
	return BoundsAccess(offset,offset):asvar()
end
function opt.InBoundsExpanded(...)
    local args = {...}
    local expand = args[#args]
    args[#args] = nil
    local min,max = List(),List()
    for i,a in ipairs(args) do
        min[i],max[i] = a - expand, a + expand
    end
    return BoundsAccess(Offset(min),Offset(max)):asvar()
end
function BoundsAccess:type() return bool end --implementing AD's API for keys


function VarDef:shift(o) return self end
function BoundsAccess:shift(o)
    return BoundsAccess(self.min:shift(o),self.max:shift(o))
end
function ImageAccess:shift(o)
    assert(Offset:isclassof(self.index), "cannot shift graph accesses!")
    return ImageAccess(self.image,self:shape(),self.index:shift(o),self.channel)
end
function ImageAccess:shape() return self._shape end -- implementing AD's API for keys

function IndexValue:shift(o)
    return IndexValue(self.dim,self.shift_ + assert(o.data[self.dim+1],"dim of index not in shift"))
end

local function shiftexp(exp,o)
    local function rename(a)
        return ad.v[a:shift(o)]
    end
    return exp:rename(rename)
end 
-------------------- More weird random stuff end

--------------------------------------- Offset start
function Offset:IsZero()
    for i,o in ipairs(self.data) do
        if o ~= 0 then return false end
    end
    return true
end
function Offset:MaxValue()
    local m = 0
    for i, o in ipairs(self.data) do
        if o > m then m = o end
    end
    return m
end
function Offset:Invert()
    local r = terralib.newlist()
    for i,o in ipairs(self.data) do
        r[i] = -o
    end
    return Offset(r)
end
function Offset:Min(rhs)
    assert(Offset:isclassof(rhs) and #self.data == #rhs.data)
    local r = List()
    for i = 1,#self.data do
        r[i] = math.min(self.data[i],rhs.data[i])
    end
    return Offset(r)
end
function Offset:Max(rhs)
    assert(Offset:isclassof(rhs) and #self.data == #rhs.data)
    local r = List()
    for i = 1,#self.data do
        r[i] = math.max(self.data[i],rhs.data[i])
    end
    return Offset(r)
end
function Offset:shift(o)
    assert(Offset:isclassof(o) and #o.data == #self.data)
    local ns = terralib.newlist()
    for i = 1,#self.data do
        ns[i] = self.data[i] + o.data[i]
    end
    return Offset(ns)
end
--------------------------------------- Offset end

----------------------------------- IndexSpace ---------------------------------------
-- TODO contains CUDA  stuff
-- TODO can't find many usages of this class in this file

function IndexSpace:cardinality()
    local c = 1
    for i,d in ipairs(self.dims) do
        c = c * d.size
    end
    return c
end

-- TODO can't find usages of this, but grepping is difficult because 'init' is a common string
function IndexSpace:init()
    self._string = self.dims:map(function(x) return x.name end):concat("_")
end

function IndexSpace:__tostring() return self._string end

function IndexSpace:ZeroOffset()
    if self._zerooffset then return self._zerooffset end
    local zeros = terralib.newlist()
    for i = 1,#self.dims do
        zeros:insert(0)
    end
    self._zerooffset = Offset(zeros)
    return self._zerooffset
end

function IndexSpace:indextype()
    if self._terratype then return self._terratype end
    local dims = self.dims
    assert(#dims > 0, "index space must have at least 1 dimension")
    local struct Index {}
    self._terratype = Index

    local params,params2 = List(),List()
    local fieldnames = List()
    for i = 1,#dims do
        local n = "d"..tostring(i-1)
        params:insert(symbol(int,n))
        params2:insert(symbol(int,n))
        fieldnames:insert(n)
        Index.entries:insert { n, int }
    end

    -- explanation: let's say X is of type index with X.d0 = 5 and X.d1 = 3. Then
    -- X(1,2) returns another (anonymous) value of type index with d0=5+1=6 and
    -- d1=3+2=5
    terra Index.metamethods.__apply(self : &Index, [params])
        var rhs : Index
        escape
            for i = 1,#dims do
                emit quote  
                    rhs.[fieldnames[i]] = self.[fieldnames[i]] + [params[i]]
                end 
            end
        end
        return rhs
    end

    -- TODO only used in following function, make local there
    -- this seems to convert an (x,y) index to a linear index
    local function genoffset(self)
        local s = 1
        local offset = `self.d0
        for i = 2,#dims do
            s = s * dims[i-1].size
            offset = `s*self.[fieldnames[i]] + offset
        end
        return offset
    end
    terra Index:tooffset()
        return [genoffset(self)]
    end

    -- this function is local to here and only used during the generation of the
    -- following terra-functions
    local function genbounds(self,bmins,bmaxs)
        local valid
        for i = 1, #dims do
            local n = fieldnames[i]
            local bmin,bmax = 0,0
            if bmins then
                bmin = assert(bmins[i])
            end
            if bmaxs then
                bmax = assert(bmaxs[i])
            end
            local v = `self.[n] >= -[bmin] and self.[n] < [dims[i].size] - [bmax]
            if valid then
                valid = `valid and v
            else
                valid = v
            end
        end
        return valid
    end
    terra Index:InBounds() return [ genbounds(self) ] end

    terra Index:InBoundsExpanded([params],[params2]) return [ genbounds(self,params,params2) ] end

    if #dims <= 3 then
        local dimnames = "xyz"
        terra Index:initFromCUDAParams() : bool -- add 'x', 'y' and 'z' field to the index
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
        print(Index) -- debug
        for k,v in pairs(Index.methods)  do print(k,v) end -- debug
    end
    return Index
end
----------------------------------- IndexSpace END ---------------------------------------

----------------------------------- DIM ---------------------------------------
-- TODO The only usage of this class I can find is near the function 'todim()'
-- in this file but todim() seems to be dead code


opt.dimensions = {1000,2000} -- usually provided by C API
opt.dimensions[0] = 1000
opt.dimensions[1] = 2000
function Dim:__tostring() return "Dim("..self.name..")" end

-- TODO who uses this? grep can't find anything
function opt.Dim(name, idx)
    idx = assert(tonumber(idx), "expected an index for this dimension")
    local size = tonumber(opt.dimensions[idx])
    return Dim(name,size,idx)
end
----------------------------------- DIM END ---------------------------------------


function opt.problemSpecFromFile(filename)
    local file, errorString = terralib.loadfile(filename)
    print('\n\n\n')
    print('START the file inside opt.problemSpecFromFile')
    print(file)
    print('END the file inside opt.problemSpecFromFile')
    print('\n\n\n')
    if not file then
        error(errorString, 0)
    end
    local P = ProblemSpec2() -- opt.ProblemSpec() is defined in this file somewhere below, seems to mostly return a ProblemSpecAD instance that is more or less uninitialized
    print('\n\n\n')
    print('START inside opt.problemSpecFromFile: result of ad.problemSpec()')
    printt(P)
    print('\n')
    print('the extraarguments')
    printt(  P.extraarguments)
    print('\n')
    print('the P')
    printt(  P.P)
    print('\n')
    print('the P.parameters')
    printt(  P.P.parameters)
    print('\n')
    print('the P.functions')
    printt(  P.P.functions)
    print('\n')
    print('the excludeexps')
    printt(  P.excludeexps)
    print('\n')
    print('the nametoimage')
    printt(  P.nametoimage)
    print('\n')
    print('the precomputed')
    printt(  P.precomputed)
    print('END inside opt.problemSpecFromFile: result of ad.problemSpec()')
    print('\n\n\n')
    print('\n\n\n')
    print('START the libinstance inside opt.problemSpecFromFile')
    local libinstance = optlib(P)
    printt(libinstance)
    print('END the libinstance inside opt.problemSpecFromFile')
    print('\n\n\n')
    setfenv(file,libinstance) -- makes e.g. Energy() etc. known to the input file but not e.g. Unknown() (where does that come from???). Answer: libinstance has an __index metamethod that looks up e.g. Dim() in opt, ad modules AND in 'P' from above, which is a ProblemSpecAD instance --> language definition spread over source code, wtf?
    print('\n\n\n')
    print('START the result inside opt.problemSpecFromFile')
    local result = file()
    print('END the result inside opt.problemSpecFromFile')
    print('\n\n\n')
    if ProblemSpec:isclassof(result) then -- NOTE: this branch is not used for image_warping.t, code seems to skip it
        return result
    end
    -- original code start
    -- return libinstance.Result() -- returns P:Cost(unpack(terms)), where terms is a list that collects everything passed to Energy(), i.e. it executes ProblemSpecAD:Cost(...)
    -- original code end

    local theterms = libinstance.getTerms() -- returns 'terms'
    local terms = extractresidualterms(unpack(theterms)) -- seems to hold 'let ... in ... end' statements that represent the residual terms
    print('\n\n\n')
    print('START Inside ProblemSpecAD:Cost(), the terms')
    -- printt(terms)
    print('END Inside ProblemSpecAD:Cost(), the terms')
    print('\n\n\n')

    
    local energyspecs = toenergyspecs(terms) -- wraps the terms inside an 'EnergySpec' object
    local functionspecs = P:Cost(energyspecs)
    return functionspecs
end

-- TODO this is only used by ProblemSpec and ProblemSpecAD, so make it a class/object method and inherit appropriately
local function toispace(ispace)
    if not IndexSpace:isclassof(ispace) then -- for handwritten API
        assert(#ispace > 0, "expected at least one dimension")
        ispace = IndexSpace(List(ispace)) 
    end
    return ispace
end

------------------------- ProblemSpecAD start
-- TODO find out how this relates to ProblemSpec, opt.ProblemSpec, etc. and make meaningful groups
function ProblemSpecAD:UsesLambda() return self.P:UsesLambda() end
function ProblemSpecAD:UsePreconditioner(v)
        self.P:UsePreconditioner(v)
end

function ProblemSpecAD:Image(name,typ,dims,idx,isunknown)
    print("START Inside ProblemSpecAD:Image(...)")
    print('\nisunknown:')
    print(isunknown)
   if not terralib.types.istype(typ) then
        typ, dims, idx, isunknown = opt_float, typ, dims, idx --shift arguments left
    end
    isunknown = isunknown and true or false
    local ispace = toispace(dims)
    assert( (type(idx) == "number" and idx >= 0) or idx == "alloc", "expected an index number") -- alloc indicates that the solver should allocate the image as an intermediate
    self.P:Image(name,typ,ispace,idx,isunknown)
    local r = Image(name,self.P:ImageType(typ,ispace),not util.isvectortype(typ),isunknown and A.UnknownLocation or A.StateLocation)
    self.nametoimage[name] = r
    print("END Inside ProblemSpecAD:Image(...)")
    return r
end
function ProblemSpecAD:Unknown(name,typ,dims,idx) 
return self:Image(name,typ,dims,idx,true) -- 'true' tells opt that this is an unknown, see also ProblemSpecAD:Image(...)
end

function ProblemSpecAD:UnknownArgument(argpos)
    if not self.extraarguments[argpos] then
        local r = {}
        for _,ip in ipairs(self.P:UnknownType().images) do
            local template = self.nametoimage[ip.name]
            r[ip.name] = Image(ip.name,template.type,template.scalar,A.ArgumentLocation(argpos))
        end
        self.extraarguments[argpos] = r
    end
    return self.extraarguments[argpos]
end

function ProblemSpecAD:ImageTemporary(name,ispace)
    self.P:Image(name,opt_float,ispace,"alloc",false)
    local r = Image(name,self.P:ImageType(opt_float,ispace),true,A.StateLocation)
    self.nametoimage[name] = r
    return r
end

function ProblemSpecAD:ImageWithName(name)
    return assert(self.nametoimage[name],"unknown image name?")
end

-- Image START -------------------------------------------------------------
-- TODO put with Image stuff

function Image:__tostring() return self.name end

-- TODO put next two with Image stuff
function Image:DimCount() return #self.type.ispace.dims end
function Image:__call(first,...)
    local index,c
    if GraphElement:isclassof(first) or Offset:isclassof(first) then
        index = first
        c = ...
    else
        local o = terralib.newlist { (assert(first,"no arguments?")) }
        for i = 1,self:DimCount() - 1 do
            o:insert((select(i,...)))
        end
        index = Offset(o)
        c = select(self:DimCount(), ...)
    end
    if GraphElement:isclassof(index) then    
        assert(index.ispace == self.type.ispace,"graph element is in a different index space from image")
    end
    c = tonumber(c)
    assert(not c or c < self.type.channelcount, "channel outside of range")
    if self.scalar or c then
        return ImageAccess(self,ad.scalar,index,c or 0):asvar()
    else
        local r = {}
        for i = 1,self.type.channelcount do
            r[i] = ImageAccess(self,ad.scalar,index,i-1):asvar()
        end
        return ad.Vector(unpack(r))
    end
end
-- Image START -------------------------------------------------------------

-- TODO find an appropriate place for this
local function MapAndGroupBy(list,fn,...)
    local groups,map = List(),{}
    for _,l in ipairs(list) do
        local g,v = fn(l,...)
        if not map[g] then
            map[g] = List()
            groups:insert(g)
        end
        map[g]:insert(v)
    end
    return groups,map
end

-- CREATE STUFF START
-- TODO find out what this does and put in some file that states the purpose
-- local function bboxforexpression(ispace,exp)
--     local usesbounds = false
--     local bmin,bmax = ispace:ZeroOffset(),ispace:ZeroOffset()
--     exp:visit(function(a)
--         if ImageAccess:isclassof(a) then
--             assert(Offset:isclassof(a.index,"bbox not defined for graphs"))
--             if a.image.gradientimages then
--                 local shiftedbbox = a.image.bbox:shift(a.index)
--                 bmin,bmax = bmin:Min(shiftedbbox.min),bmax:Max(shiftedbbox.max)
--             else
--                 bmin,bmax = bmin:Min(a.index),bmax:Max(a.index)
--             end
--         elseif BoundsAccess:isclassof(a) then
--             usesbounds = true
--         end
--     end)
--     if usesbounds then 
--         bmin,bmax = ispace:ZeroOffset(),ispace:ZeroOffset()
--     end
--     return BoundsAccess(bmin,bmax)
-- end

-- local function classifyexpression(exp) -- what index space, or graph is this thing mapped over
--     local classification
--     local seenunknown = {}
--     local unknownaccesses = terralib.newlist()
--     local function addunknown(u)
--         if not seenunknown[u] then
--             unknownaccesses:insert(u)
--             seenunknown[u] = true
--         end
--     end
--     exp:visit(function(a)
--         if ImageAccess:isclassof(a) then -- assume image X is unknown
--             if a.image.location == A.UnknownLocation then
--                 addunknown(a)
--             elseif a.image.gradientimages then
--                 for i,im in ipairs(a.image.gradientimages) do
--                     assert(Offset:isclassof(a.index),"NYI - precomputed with graphs")
--                     addunknown(im.unknown:shift(a.index))
--                 end
--             end
--             local aclass = Offset:isclassof(a.index) and A.CenteredFunction(a.image.type.ispace) or A.GraphFunction(a.index.graph.name)
--             assert(nil == classification or aclass == classification, "residual contains image reads from multiple domains")
--             classification = aclass
--         end
--     end)
--     local template = A.ResidualTemplate(exp,unknownaccesses)
--     if not classification then
--         error("residual must actually use some image")
--     end
--     if classification.kind == "CenteredFunction" then
--         exp:visit(function(a)
--             if BoundsAccess:isclassof(a) and #a.min.data ~= #classification.ispace.dims then
--                 error(string.format("%s does not match index space %s",a,classification.ispace.dims))
--             end
--         end)
--         -- by default zero-out any residual computation that uses out-of-bounds things
--         -- users can opt for per-residual custom behavior using the InBounds checks
--         local bbox = bboxforexpression(classification.ispace,exp)
--         template.expression = ad.select(bbox:asvar(),exp,0)
--     end
--     return classification,template
-- end

-- -- TODO Important function, move downwards
-- local function toenergyspecs(Rs)    
--     local kinds,kind_to_templates = MapAndGroupBy(Rs,classifyexpression)
--     return kinds:map(function(k) return A.EnergySpec(k,kind_to_templates[k]) end)
-- end

--given that the residual at (0,0) uses the variables in 'unknownsupport',
--what is the set of residuals will use variable X(0,0).
--this amounts to taking each variable in unknown support and asking which residual is it
--that makes that variable X(0,0)
-- TODO only used in create***(), so make local there after those functions have been grouped in a file
local function residualsincludingX00(unknownsupport,unknown,channel)
    assert(channel)
    local r = terralib.newlist()
    for i,u in ipairs(unknownsupport) do
        assert(Offset:isclassof(u.index),"unexpected graph access")
        if u.image == unknown and u.channel == channel then
            r:insert(u.index:Invert())
        end
    end
    return r
end

-- TODO only used in create***(), so make local there after those functions have been grouped in a file
local function unknownsforresidual(r,unknownsupport)
    return unknownsupport:map("shift",r)
end

-- TODO only used in create***(), so make local there after those functions have been grouped in a file
local function createzerolist(N)
    local r = terralib.newlist()
    for i = 1,N do
        r[i] = ad.toexp(0)
    end
    return r
end
    
-- TODO only used in create***(), so make local there after those functions have been grouped in a file
local function lprintf(ident,fmt,...)
    if true then return end 
    local str = fmt:format(...)
    ident = (" "):rep(ident*4)
    str = ident..str:gsub('\n', "\n"..ident)
    return print(str) 
end
local EMPTY = List()
local function createjtjcentered(PS,ES)
    local UnknownType = PS.P:UnknownType()
    local ispace = ES.kind.ispace
    local N = UnknownType:VectorSizeForIndexSpace(ES.kind.ispace)
    local P = PS:UnknownArgument(1)
    local CtC = PS:UnknownArgument(2)
    --local Pre = PS:UnknownArgument(3)
    local P_hat_c = {}
    local conditions = terralib.newlist()
    for rn,residual in ipairs(ES.residuals) do
        local F,unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"\n\n\n\n\n##################################################")
        lprintf(0,"r%d = %s",rn,F)
        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do 
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(ispace:ZeroOffset(),chan)
            local residuals = residualsincludingX00(unknownsupport,unknown,chan)
            for _,r in ipairs(residuals) do
                local rexp = shiftexp(F,r)
                local condition,drdx00 = ad.splitcondition(rexp:d(x))
                lprintf(1,"instance:\ndr%d_%s/dx00[%d] = %s",rn,tostring(r),chan,tostring(drdx00))
                local unknowns = unknownsforresidual(r,unknownsupport)
                for _,u in ipairs(unknowns) do
                    local uv = ad.v[u]
                    local condition2, drdx_u = ad.splitcondition(rexp:d(uv))
                    local exp = drdx00*drdx_u

                    lprintf(2,"term:\ndr%d_%s/dx%s[%d] = %s",rn,tostring(r),tostring(u.index),u.chan,tostring(drdx_u))
                    local conditionmerged = condition*condition2
                    if not P_hat_c[conditionmerged] then
                        conditions:insert(conditionmerged)
                        P_hat_c[conditionmerged] = createzerolist(N)
                    end
                    P_hat_c[conditionmerged][idx+1] = P_hat_c[conditionmerged][idx+1] + P[u.image.name](u.index,u.channel)*exp
                end
            end
        end
    end
    local P_hat = createzerolist(N)
    for _,c in ipairs(conditions) do
        for i = 1,N do
            P_hat[i] = P_hat[i] + c*P_hat_c[c][i]
        end
    end
    for i,p in ipairs(P_hat) do
        P_hat[i] = 1.0 * p
    end
    if PS:UsesLambda() then
        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local u = unknown(ispace:ZeroOffset(),chan)
            P_hat[idx+1] = P_hat[idx+1] + CtC[unknownname](ispace:ZeroOffset(),chan)*P[unknownname](ispace:ZeroOffset(),chan)
        end
    end
    dprint("JTJ[nopoly] = ", ad.tostrings(P_hat))
    P_hat = ad.polysimplify(P_hat)
    dprint("JTJ[poly] = ", ad.tostrings(P_hat))
    local r = ad.Vector(unpack(P_hat))
    local result = A.FunctionSpec(ES.kind,"applyJTJ", List {"P", "CtC"}, List{r}, EMPTY,ES)
    return result
end


local function createjtjgraph(PS,ES)
    local P,Ap_X = PS:UnknownArgument(1),PS:UnknownArgument(2)

    local result = ad.toexp(0)
    local scatters = List() 
    local scattermap = {}
    local function addscatter(u,exp)
        local s = scattermap[u]
        if not s then
            s =  Scatter(Ap_X[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        local Jp = ad.toexp(0)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            Jp = Jp + partial*P[u.image.name](u.index,u.channel)
        end
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local jtjp = 1.0 * Jp*partial
            result = result + P[u.image.name](u.index,u.channel)*jtjp
            addscatter(u,jtjp)
        end
    end

    return A.FunctionSpec(ES.kind,"applyJTJ", List {"P", "Ap_X"}, List { result }, scatters, ES)
end


local function createjtfcentered(PS,ES)
   local UnknownType = PS.P:UnknownType()
   local ispace = ES.kind.ispace
   local N = UnknownType:VectorSizeForIndexSpace(ispace)

   local F_hat = createzerolist(N) --gradient
   local P_hat = createzerolist(N) --preconditioner
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))

        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(ispace:ZeroOffset(),chan)
            
            local residuals = residualsincludingX00(unknownsupport,unknown,chan)

            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f)
                local dfdx00 = F_x:d(x)		-- entry of J^T
                local dfdx00F = dfdx00*F_x	-- entry of \gradF == J^TF
                F_hat[idx+1] = F_hat[idx+1] + dfdx00F			-- summing it up to get \gradF

                local dfdx00Sq = dfdx00*dfdx00	-- entry of Diag(J^TJ)
                P_hat[idx+1] = P_hat[idx+1] + dfdx00Sq			-- summing the pre-conditioner up
                lprintf(2,"dR[%d]_%s/dx[%d] = %s",ridx,tostring(f),chan,tostring(dfdx00F))
            end

        end
    end
	for i = 1,N do
	    if not PS.P.usepreconditioner then
		    P_hat[i] = ad.toexp(1.0)
	    else
		    P_hat[i] = ad.polysimplify(P_hat[i])
	    end
	    F_hat[i] = ad.polysimplify(1.0 * F_hat[i])
	end
	dprint("JTF =", ad.tostrings({F_hat[1], F_hat[2], F_hat[3]}))
    return A.FunctionSpec(ES.kind,"evalJTF", EMPTY, List{ ad.Vector(unpack(F_hat)), ad.Vector(unpack(P_hat)) }, EMPTY,ES)
end

local function createmodelcost(PS,ES)
    local UnknownType = PS.P:UnknownType()
    local ispace = ES.kind.ispace
    local N = UnknownType:VectorSizeForIndexSpace(ispace)


    local Delta = PS:UnknownArgument(1)
    local result = 0.0 --model residuals squared (and summed among unknowns...)
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)

        local JTdelta = 0.0

        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            local delta = Delta[u.image.name](u.index,u.channel)
            JTdelta = JTdelta + (partial * delta)
        end
        local residual_m = F + JTdelta
        result = result + (residual_m*residual_m)
    end
    result = ad.polysimplify(0.5*result)
    return A.FunctionSpec(ES.kind,"modelcost", List {"Delta"}, List{ result }, EMPTY,ES)
end

local function createmodelcostgraph(PS,ES)
    local Delta = PS:UnknownArgument(1)
    local result = 0.0 --model residuals squared (and summed among unknowns...)
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)

        local JTdelta = 0.0

        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            local delta = Delta[u.image.name](u.index,u.channel)
            JTdelta = JTdelta + (partial * delta)
        end
        local residual_m = F + JTdelta
        result = result + (residual_m*residual_m)
    end
    result = ad.polysimplify(0.5*result)
    return A.FunctionSpec(ES.kind, "modelcost", List { "Delta" }, List{ result }, EMPTY,ES)
end


local function createjtfgraph(PS,ES)
    local R,Pre = PS:UnknownArgument(1),PS:UnknownArgument(2)
    local scatters = List() 
    local scattermap = { [R] = {}, [Pre] = {}}
    local function addscatter(im,u,exp)
        local s = scattermap[im][u]
        if not s then
            s =  Scatter(im[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[im][u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            addscatter(R,u,-1.0*partial*F)
            addscatter(Pre,u,partial*partial)
        end
    end
    return A.FunctionSpec(ES.kind, "evalJTF", List { "R", "Pre" }, EMPTY, scatters,ES)
end

local function computeCtCcentered(PS,ES)
   local UnknownType = PS.P:UnknownType()
   local ispace = ES.kind.ispace
   local N = UnknownType:VectorSizeForIndexSpace(ispace)
   local D_hat = createzerolist(N) --gradient
    
    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))

        for idx,unknownname,chan in UnknownType:UnknownIteratorForIndexSpace(ispace) do
            local unknown = PS:ImageWithName(unknownname) 
            local x = unknown(ispace:ZeroOffset(),chan)

            local residuals = residualsincludingX00(unknownsupport,unknown,chan)
            local sum = 0
            for _,f in ipairs(residuals) do
                local F_x = shiftexp(F,f)
                local dfdx00 = F_x:d(x)     -- entry of J^T
                local dfdx00Sq = dfdx00*dfdx00  -- entry of Diag(J^TJ)

                local inv_radius = 1.0 / PS.trust_region_radius
                local D_entry = dfdx00Sq*inv_radius 
                D_hat[idx+1] = D_hat[idx+1] + D_entry
            end

        end
    end
    for i = 1,N do
        D_hat[i] = ad.polysimplify(D_hat[i])
    end
    return A.FunctionSpec(ES.kind,"computeCtC", List { }, List{ ad.Vector(unpack(D_hat)) }, EMPTY,ES)
end

local function computeCtCgraph(PS,ES)
    local CtC = PS:UnknownArgument(1),PS:UnknownArgument(2)
    local scatters = List() 
    local scattermap = { [CtC] = {}}

    local function addscatter(im,u,exp)
        local s = scattermap[im][u]
        if not s then
            s =  Scatter(im[u.image.name],u.index,u.channel,ad.toexp(0),"add")
            scattermap[im][u] = s
            scatters:insert(s)
        end
        s.expression = s.expression + exp
    end
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        local unknownvars = unknownsupport:map(function(x) return ad.v[x] end)
        local partials = F:gradient(unknownvars)
        for i,partial in ipairs(partials) do
            local u = unknownsupport[i]
            assert(GraphElement:isclassof(u.index))
            local inv_radius = 1.0 / PS.trust_region_radius
            addscatter(CtC,u,partial*partial*inv_radius)
        end
    end
    return A.FunctionSpec(ES.kind, "computeCtC", List { "CtC" }, EMPTY, scatters, ES)
end

local function createdumpjcentered(PS,ES)
   local UnknownType = PS.P:UnknownType()
   local ispace = ES.kind.ispace
   local N = UnknownType:VectorSizeForIndexSpace(ispace)

    local outputs = List{}

    for ridx,residual in ipairs(ES.residuals) do
        local F, unknownsupport = residual.expression,residual.unknowns
        lprintf(0,"-------------")
        lprintf(1,"R[%d] = %s",ridx,tostring(F))
        for i,unknown in ipairs(unknownsupport) do
            outputs:insert(F:d(ad.v[unknown]))
        end
    end
    return A.FunctionSpec(ES.kind,"dumpJ", EMPTY, outputs, EMPTY,ES)
end
local function createdumpjgraph(PS,ES)
    local outputs = List{}
    for i,term in ipairs(ES.residuals) do
        local F,unknownsupport = term.expression,term.unknowns
        for i,unknown in ipairs(unknownsupport) do
            outputs:insert(F:d(ad.v[unknown]))
        end
    end
    return A.FunctionSpec(ES.kind, "dumpJ", EMPTY, outputs, EMPTY,ES)
end

-- TODO put next two with other helper functions
-- TODO only used in function below, so try to make local (it seems that this variable together with the function is supposed to behave like an object, so may this needs to be resolved differently
local lastTime = nil
-- TODO who is using this??? grep doesn't find anything
function timeSinceLast(name)
    local currentTime = terralib.currenttimeinseconds()
    if (lastTime) then
        local deltaTime = currentTime-lastTime
        print(string.format("%s: %f sec\n",name,deltaTime))
    end
    lastTime = currentTime
end

local function creategradient(unknown,costexp)
    local unknownvars = unknowns(costexp)
    local gradient = costexp:gradient(unknownvars)

    dprint("grad expression")
    local names = table.concat(unknownvars:map(function(v) return tostring(v:key()) end),", ")
    dprint(names.." = "..ad.tostrings(gradient))
    
    local gradientsgathered = createzerolist(unknown.type.channelcount)
    for i,u in ipairs(unknownvars) do
        local a = u:key()
        local shift = shiftexp(gradient[i],a.index:Invert())
        gradientsgathered[a.channel+1] = gradientsgathered[a.channel+1] + shift
    end
    dprint("grad gather")
    dprint(ad.tostrings(gradientsgathered))
    return ad.Vector(gradientsgathered)
end
    
local function createcost(ES)
    local function sumsquared(terms)
        local sum = ad.toexp(0)
        for i,t in ipairs(terms) do
            sum = sum + t*t
        end
        return 0.5*sum
    end
    local exp = sumsquared(ES.residuals:map("expression"))
    return A.FunctionSpec(ES.kind,"cost", EMPTY, List{exp}, EMPTY,ES) 
end

function createprecomputed(self,precomputedimages)

    local ispaces,image_map = MapAndGroupBy(precomputedimages,function(im) return im.type.ispace,im end)

    local precomputes = List()
    for _,ispace in ipairs(ispaces) do
        local scatters = List()
        local zoff = ispace:ZeroOffset()
        for _,im in ipairs(image_map[ispace]) do
            local expression = ad.polysimplify(im.expression)
            scatters:insert(Scatter(im, zoff, 0, im.expression, "set"))
            for _,gim in ipairs(im.gradientimages) do
                local gradientexpression = ad.polysimplify(gim.expression)
                if not ad.Const:isclassof(gradientexpression) then
                    scatters:insert(Scatter(gim.image, zoff, 0, gradientexpression, "set"))
                end
            end
        end
        local pc = A.FunctionSpec(A.CenteredFunction(ispace),"precompute", EMPTY, EMPTY, scatters)
        precomputes:insert(pc)
    end
    return precomputes
end

-- local function extractresidualterms(...)
--     local exp = terralib.newlist {}
--     for i = 1, select("#",...) do
--         local e = select(i,...)
--         if ad.ExpVector:isclassof(e) then
--             for i,t in ipairs(e:expressions()) do
--                 t = assert(ad.toexp(t), "expected an ad expression")
--                 exp:insert(t)
--             end
--         else
--             exp:insert((assert(ad.toexp(e), "expected an ad expression")))
--         end
--     end
--     return exp
-- end
-- CREATE STUFF END

-- TODO put with other ProblemSpecAD stuff
function ProblemSpecAD:Cost(energyspecs)
    -- local terms = extractresidualterms(...) -- seems to hold 'let ... in ... end' statements that represent the residual terms
    -- print('\n\n\n')
    -- print('START Inside ProblemSpecAD:Cost(), the terms')
    -- -- printt(terms)
    -- print('END Inside ProblemSpecAD:Cost(), the terms')
    -- print('\n\n\n')

    
    local functionspecs = List()
    -- local energyspecs = toenergyspecs(terms) -- wraps the terms inside an 'EnergySpec' object
    print('\n\n\n')
    print('START Inside ProblemSpecAD:Cost(), the energyspecs')
    printt(energyspecs)
    print('END Inside ProblemSpecAD:Cost(), the energyspecs')
    print('\n\n\n')
    for _,energyspec in ipairs(energyspecs) do
        functionspecs:insert(createcost(energyspec))          
        if energyspec.kind.kind == "CenteredFunction" then
            functionspecs:insert(createjtjcentered(self,energyspec))
            functionspecs:insert(createjtfcentered(self,energyspec))
            functionspecs:insert(createdumpjcentered(self,energyspec))
            
            if self.P:UsesLambda() then
                functionspecs:insert(computeCtCcentered(self,energyspec))
                functionspecs:insert(createmodelcost(self,energyspec))
            end
        else
            functionspecs:insert(createjtjgraph(self,energyspec))
            functionspecs:insert(createjtfgraph(self,energyspec))
            functionspecs:insert(createdumpjgraph(self,energyspec))
            
            if self.P:UsesLambda() then
                functionspecs:insert(computeCtCgraph(self,energyspec))
                functionspecs:insert(createmodelcostgraph(self,energyspec))         
            end
        end
    end
    functionspecs:insertall(createprecomputed(self,self.precomputed))
    for i,exclude in ipairs(self.excludeexps) do
        local class = classifyexpression(exclude)
        functionspecs:insert(A.FunctionSpec(class, "exclude", EMPTY,List{exclude}, EMPTY))
    end
    print('\n\n\n')
    print('START Inside ProblemSpecAD:Cost(), the functionspecs')
    -- functionspecs holds a FunctionSpec object for each function (e.g. applyJtJ) with the following fields (only partial list)
    -- name = applyJtJ
    -- result = let .. in ... end
    -- arguments = {...}
    printt(functionspecs)
    print('END Inside ProblemSpecAD:Cost(), the functionspecs')
    print('\n\n\n')
    
    -- original code start
    -- self:AddFunctions(functionspecs) -- turns functionspecs into proper terra functions and adds them to self.P
    -- self.P.energyspecs = energyspecs
    -- return self.P
    -- original code end
    return functionspecs
end
function VarDef:asvar() return ad.v[self] end

function ProblemSpecAD:ComputedImage(name,dims,exp)
    if ad.ExpVector:isclassof(exp) then
        local imgs = terralib.newlist()
        for i,e in ipairs(exp:expressions()) do
            imgs:insert(self:ComputedImage(name.."_"..tostring(i-1),dims,e))
        end
        return ImageVector(imgs)
    end
    exp = assert(ad.toexp(exp),"expected a math expression")
    local unknowns = terralib.newlist()
    local seen = {}
    exp:visit(function(a)
        if ImageAccess:isclassof(a) and a.image.location == A.UnknownLocation then
            assert(Offset:isclassof(a.index),"NYI - support for graphs")
            if not seen[a] then
                seen[a] = true
                unknowns:insert(a)
            end
        end
    end)
    local ispace = toispace(dims)
    local im = self:ImageTemporary(name,ispace)
    local gradients = exp:gradient(unknowns:map(function(x) return ad.v[x] end))
    im.gradientimages = terralib.newlist()
    for i,g in ipairs(gradients) do
        local u = unknowns[i]
        local gim = self:ImageTemporary(name.."_d_"..tostring(u),ispace)
        im.gradientimages:insert(GradientImage(u,g,gim))
    end
    im.expression = exp
    im.bbox = bboxforexpression(ispace,exp)
    self.precomputed:insert(im)
    return im
end

-- TODO put with Graph stuff
Graph = {}
function Graph:__tostring() return self.name end

function ProblemSpecAD:Graph(name,idx,...)
    self.P:Graph(name,idx,...)
    local g = Graph(name)
    for i = 1, select("#",...),3 do
        local name,dims,didx = select(i,...)
        local ge = GraphElement(g,name) 
        ge.ispace = toispace(dims)
        g[name] = ge
    end
    return g
end

function ProblemSpecAD:Param(name,typ,idx)
    self.P:Param(name,typ,idx)
    return ParamValue(name,typ):asvar()
end

-- TODO move to ProblemSpecAD stuff
function ProblemSpecAD:AddFunctions(functionspecs) -- takes fspecs, compiles them and stores compiled functions in self.P.functions
    local kind_to_functionmap = {}
    local kinds = List()
    for i,fs in ipairs(functionspecs) do -- group by unique function kind to pass to ProblemSpec:Functions call
        local fm = kind_to_functionmap[fs.kind]
        if not fm then
            fm = {}
            kind_to_functionmap[fs.kind] = fm
            kinds:insert(fs.kind)
        end
        assert(not fm[fs.name],"function already defined!")
        fm[fs.name] = self:CompileFunctionSpec(fs) -- takes a FunctionSpec and turns it into a "functionmap"
        if fm.derivedfrom and fs.derivedfrom then
            assert(fm.derivedfrom == fs.derivedfrom, "not same energy spec?")
        end
        fm.derivedfrom = fm.derivedfrom or fs.derivedfrom
    end
    for _,k in ipairs(kinds) do
        local fm = kind_to_functionmap[k]
        self.P:Functions(k,fm)
    end
end
------------------------- ProblemSpecAD end


------------------------- ProblemSpec START
function ProblemSpec:UsesLambda() return c.problemkind:match("LM") ~= nil end

function ProblemSpec:Image(name,typ,ispace,idx,isunknown)
    self:Stage "inputs"
    isunknown = isunknown and true or false
    self:newparameter(ImageParam(self:ImageType(typ,toispace(ispace)),isunknown,name,idx))
end

-- TODO only used from within ProblemSpec, make private somehow
function ProblemSpec:newparameter(p) -- adds new parameter
    assert(ProblemParam:isclassof(p))
    self:registername(p.name)
    self.parameters:insert(p)
end

-- TODO only used once and within ProblemSpec, make private somehow
function ProblemSpec:registername(name)
    assert(not self.names[name],string.format("name %s already in use",name))
    self.names[name] = #self.parameters + 1
end

-- TODO make this local to the following function (ProblemSpec:ImageType())
local function tovalidimagetype(typ)
    if not terralib.types.istype(typ) then return nil end
    if util.isvectortype(typ) then
        return typ.metamethods.type, typ.metamethods.N
    elseif typ:isarithmetic() then
        return typ, 1
    end
end

-- TODO move to ProblemSpec stuff
-- TODO who is using this??? grep doesn't find anything --> used e.g. in ProblemSpecAD:Image(...)
function ProblemSpec:ImageType(typ,ispace)
    local scalartype,channelcount = tovalidimagetype(typ,"expected a number or an array of numbers")
    assert(scalartype,"expected a number or an array of numbers")
    return ImageType(ispace,scalartype,channelcount) 
end

-- TODO only called from within ProblemSpec, make private somehow
--  this functions provides a mechanism to ensure that certain functions can only be called in a certain order
function ProblemSpec:Stage(name) -- sets the stage to new stage and ensures that stages only move forward
    assert(PROBLEM_STAGES[self.stage] <= PROBLEM_STAGES[name], "all inputs must be specified before functions are added")
    self.stage = name
end

-- somehow builds a list (images) of unknowns and wrappes them in 'UnknownType'
function ProblemSpec:UnknownType() -- TODO what does this do???
    self:Stage "functions"
    if not self._UnknownType then
        local images = List()
        for _,p in ipairs(self.parameters) do
            if p.isunknown then images:insert(p) end
        end
        self._UnknownType = UnknownType(images)
    end
    return self._UnknownType
end
------------------------- ProblemSpec END

function ProblemSpec2()
    local ps = ProblemSpecAD()
    ps.P,ps.nametoimage,ps.precomputed,ps.extraarguments,ps.excludeexps = opt.ProblemSpec(), {}, List(), List(), List()
    if ps.P:UsesLambda() then
        ps.trust_region_radius = ps:Param("trust_region_radius",opt_float,-1)
        ps.radius_decrease_factor = ps:Param("radius_decrease_factor",opt_float,-1)
        ps.min_lm_diagonal = ps:Param("min_lm_diagonal",opt_float,-1)
        ps.max_lm_diagonal = ps:Param("max_lm_diagonal",opt_float,-1)
    end
    return ps
end

function opt.ProblemSpec()
    local ps = ProblemSpec()
    ps.parameters = terralib.newlist() -- ProblemParam*
    ps.names = {} -- name -> index in parameters list
    ps.functions = List() -- ProblemFunctions*
    ps.maxStencil = 0
    ps.stage = "inputs"
    ps.usepreconditioner = false
    ps.problemkind = opt.problemkind
    return ps
end

opt.problemSpecFromFile('testinput.t')
