TRACE = true -- TODO added by SO to turn on/off lots of print-statements required to understand how the code works
function printt(thing)
  print('the name of the thing:')
  if TRACE then
    print(thing)
  end
  print('the thing as table:')
  for k,v in pairs(thing) do print(k,v) end
end

local N = opt.Dim("N",0)
local NUMEDGES = opt.Dim("NUMEDGES", 1)


local w_fitSqrt =   Param("w_fitSqrt", float, 0)
local w_regSqrt =   Param("w_regSqrt", float, 1)
local Offset =      Unknown("Offset", opt_float3,{N},2)            --vertex.xyz, rotation.xyz <- unknown
local Angle = 	    Unknown("Angle",opt_float3,{N},3)	
local UrShape =     Array("UrShape",opt_float3,{N},4)        --original position: vertex.xyz
local Constraints = Array("Constraints",opt_float3,{N},5)    --user constraints
-- local G = Graph("G", 6, "v0", {N}, 7, "v1", {N}, 8)
local G = Graph("G", {NUMEDGES}, "v0", {N}, 7, "v1", {N}, 8)
UsePreconditioner(true)

print('\n\n\n')
print('START Inside arap_mesh_deformation.t')
print('\nThe Offset(G.v0):')
printt(Offset(G.v0))
print('\nThe Offset(G.v0).data:')
printt(Offset(G.v0).data)
print('\nThe Offset(G.v0).data[1]:')
printt(Offset(G.v0).data[1])
print('\nThe Offset(G.v0).data[1].key_ (ImageAccess)(this is "a" in classifyexpression() ):')
printt(Offset(G.v0).data[1].key_)
print('\nThe Offset(G.v0).data[1].key_.index:')
printt(Offset(G.v0).data[1].key_.index)

print('END Inside arap_mesh_deformation.t')
print('\n\n\n')

--fitting
local e_fit = Offset(0) - Constraints(0)
local valid = greatereq(Constraints(0,0), -999999.9)
Energy(Select(valid,w_fitSqrt*e_fit,0))

--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) - Rotate3D(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))
Energy(w_regSqrt*ARAPCost)
