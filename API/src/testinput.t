local W,H = Dim("W",0), Dim("H",1)
local X = Unknown("X", opt_float,{W,H},0) --original mesh position
local A = Array("A", opt_float,{W,H},1) -- user constraints

print('inside testinput.t')

-- w_fit, w_reg = .1, .9
-- Energy(w_fit*(X(0,0) - A(0,0)),
--        w_reg*(X(0,0) - X(1,0)),
--        w_reg*(X(0,0) - X(0,1)))
-- Energy(X(0,0) - A(0,0))
Energy(A(0,0))
-- print(X)
-- print(A)
for k,v in pairs(X) do print(k,v) end
for k,v in pairs(A) do print(k,v) end
