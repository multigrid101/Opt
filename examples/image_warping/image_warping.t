
TRACE = true -- TODO added by SO to turn on/off lots of print-statements required to understand how the code works
function printt(thing)
  print('the name of the thing:')
  if TRACE then
    print(thing)
  end
  print('the thing as table:')
  for k,v in pairs(thing) do print(k,v) end
end

local W,H = Dim("W",0), Dim("H",1)

print('BEFORE Offset ctor call')
local Offset = Unknown("Offset",opt_float2,{W,H},0)
print('AFTER Offset ctor call')

local Angle = Unknown("Angle",opt_float,{W,H},1)			

print('BEFORE UrShape ctor call')
local UrShape = Array("UrShape", opt_float2,{W,H},2) --original mesh position
print('AFTER UrShape ctor call')

local Constraints = Array("Constraints", opt_float2,{W,H},3) -- user constraints
local Mask = Array("Mask", opt_float, {W,H},4) -- validity mask for mesh
local w_fitSqrt = Param("w_fitSqrt", float, 5)
local w_regSqrt = Param("w_regSqrt", float, 6)

UsePreconditioner(true)
Exclude(Not(eq(Mask(0,0),0)))

--regularization
for x,y in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local e_reg = w_regSqrt*((Offset(0,0) - Offset(x,y)) 
                             - Rotate2D(Angle(0,0),(UrShape(0,0) - UrShape(x,y))))          
    local valid = InBounds(x,y) * eq(Mask(x,y),0) * eq(Mask(0,0),0)
    Energy(Select(valid,e_reg,0))
end

print('\n\n\n')
print('START inside image_warping.t')
print('\n')
print('The Dim W')
printt(W)
print('\n')
print('The Offset(Unknown)')
printt(Offset)
print('\n')
print('The Unknown.location')
printt(Offset.location)
print('\n')
print('The Unknown.metamethods')
printt(UrShape.__index)
print('\n')
print('The UrShape(Array)')
printt(UrShape)
print('\n')
print('The Array.location')
printt(UrShape.location)
print('\n')
print('The indexed Urshape Urshape(2,3)')
printt(UrShape(2,3))
print('\n')
print('The indexed Urshape Urshape(2,3).data')
printt(UrShape(2,3).data)
print('\n')
print('The indexed Urshape Urshape(2,3).data[1]')
printt(UrShape(2,3).data[1])
print('\n')
print('The indexed Urshape Urshape(2,3).data[1].key_')
printt(UrShape(2,3).data[1].key_)
print('\n')
print('The indexed Urshape Urshape(2,3).data[1].key_.image')
printt(UrShape(2,3).data[1].key_.image)
print('\n')
print('The indexed Urshape Urshape(2,3).data[1].key_.index')
printt(UrShape(2,3).data[1].key_.index)
print('\n')
print('The indexed Urshape Urshape(2,3).data[1].key_.index.data')
printt(UrShape(2,3).data[1].key_.index.data)
print('\n')
print('The Urshape(2,3)-Urshape(1,4)')
printt((UrShape(2,3)-UrShape(1,4)))
print('END inside image_warping.t')
print('\n\n\n')

--fitting
local e_fit = (Offset(0,0)- Constraints(0,0))
local valid = All(greatereq(Constraints(0,0),0))
Energy(w_fitSqrt*Select(valid, e_fit , 0.0))
