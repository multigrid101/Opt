-- just run this file with terra to do unit-tests on the linalg stuff

local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
]]

la = require('linalg_cpu')
tu = require('tunit')

-- local terra computeNnzPatternATA(handle : &opaque, -- needed by cusparse lib TODO refactor
--                                 descr : &opaque, -- needed by cusparse lib TODO refactor
--                                 nUnknowns : int, -- if A is nxm, then this is m
--                                 nResiduals : int, -- if A is nxm, then this is n
--                                 nnzA : int,
--                                 rowPtrA : &int, colIndA : &int,
--                                 rowPtrATA : &int, ptrTo_colIndATA : &&int, nnzATAptr : &int)


terra computeNnzPatternAT_test()
  var nRowsA = 3
  var nColsA = 4

  var nRowsAT = nColsA
  var nColsAT = nRowsA

  var nRowsATA = nColsA
  var nColsATA = nColsA

  var nnzA = 7
  var nnzAT = nnzA
  var nnzATA : int

  var rowPtrA = [&int](C.malloc( (nRowsA+1) * sizeof(int) ))
  var colIndA = [&int](C.malloc( nnzA * sizeof(int) ))
  var valA = [&float](C.malloc( nnzA * sizeof(float) ))

  var rowPtrAT = [&int](C.malloc( (nRowsAT+1) * sizeof(int) ))
  var colIndAT = [&int](C.malloc( nnzAT * sizeof(int) ))
  var valAT = [&float](C.malloc( nnzAT * sizeof(float) ))

  var rowPtrATA = [&int](C.malloc( (nRowsATA+1) * sizeof(int) ))
  var colIndATA : &int
  var valATA : &float

  var x = [&float](C.malloc( nColsA * sizeof(float) ))
  var y = [&float](C.malloc( nRowsA * sizeof(float) ))

  rowPtrA[0] = 0
  rowPtrA[1] = 3
  rowPtrA[2] = 5
  rowPtrA[3] = 7

  colIndA[0] = 0
  colIndA[1] = 2
  colIndA[2] = 3
  colIndA[3] = 1
  colIndA[4] = 3
  colIndA[5] = 0
  colIndA[6] = 2

  valA[0] = 1
  valA[1] = 3
  valA[2] = 4
  valA[3] = 9
  valA[4] = 2
  valA[5] = 7
  valA[6] = 8

  x[0] = 3
  x[1] = 5
  x[2] = 7
  x[3] = 9

  la.computeNnzPatternAT(nil, nil, nColsA, nRowsA, nnzA,
                         rowPtrA, colIndA,
                         rowPtrAT, colIndAT)

  la.computeAT(nil, nil, nColsA, nRowsA, nnzA,
                         valA, rowPtrA, colIndA,
                         valAT, rowPtrAT, colIndAT)

  la.applyAtoVector(nil, nil,
                  nColsA, nRowsA, nnzA,
                  valA, rowPtrA , colIndA,
                  x,y)

  tu.assert(rowPtrAT[0] == 0)
  tu.assert(rowPtrAT[1] == 2)
  tu.assert(rowPtrAT[2] == 3)
  tu.assert(rowPtrAT[3] == 5)
  tu.assert(rowPtrAT[4] == 7)

  tu.assert(colIndAT[0] == 0)
  tu.assert(colIndAT[1] == 2)
  tu.assert(colIndAT[2] == 1)
  tu.assert(colIndAT[3] == 0)
  tu.assert(colIndAT[4] == 2)
  tu.assert(colIndAT[5] == 0)
  tu.assert(colIndAT[6] == 1)

  tu.assert(valAT[0] == 1)
  tu.assert(valAT[1] == 7)
  tu.assert(valAT[2] == 9)
  tu.assert(valAT[3] == 3)
  tu.assert(valAT[4] == 8)
  tu.assert(valAT[5] == 4)
  tu.assert(valAT[6] == 2)

  tu.assert(y[0] == 60)
  tu.assert(y[1] == 63)
  tu.assert(y[2] == 77)

  la.computeNnzPatternATA(nil, nil, nColsA, nRowsA, nnzA,
                          rowPtrA, colIndA,
                          rowPtrATA, &colIndATA, &nnzATA)


  valATA = [&float](C.malloc( nnzATA * sizeof(float) ))
  la.computeATA(nil, nil,
                nColsA, nRowsA, nnzA, nnzATA,
                valA, rowPtrA, colIndA,
                valAT, rowPtrAT, colIndAT,
                valATA, rowPtrATA, colIndATA)

  -- la.printNnzPatternA(nRowsA, nColsA, nnzA,
  --                     rowPtrA, colIndA)
  -- la.printValuesA(nRowsA, nColsA, nnzA,
  --                     rowPtrA, colIndA, valA)

  -- la.printNnzPatternA(nRowsAT, nColsAT, nnzAT,
  --                     rowPtrAT, colIndAT)
  -- la.printValuesA(nRowsAT, nColsAT, nnzAT,
  --                     rowPtrAT, colIndAT, valAT)

  -- la.printNnzPatternA(nRowsATA, nColsATA, nnzATA,
  --                     rowPtrATA, colIndATA)
  -- la.printValuesA(nRowsATA, nColsATA, nnzATA,
  --                     rowPtrATA, colIndATA, valATA)

  -- C.printf('The vals:\n')
  -- for k = 0,nnzATA do
  --   C.printf('%f\n', valATA[k])
  -- end

  tu.assert(rowPtrATA[0] == 0)
  tu.assert(rowPtrATA[1] == 3)
  tu.assert(rowPtrATA[2] == 5)
  tu.assert(rowPtrATA[3] == 8)
  tu.assert(rowPtrATA[4] == 12)

  tu.assert(colIndATA[0] == 0)
  tu.assert(colIndATA[1] == 2)
  tu.assert(colIndATA[2] == 3)
  tu.assert(colIndATA[3] == 1)
  tu.assert(colIndATA[4] == 3)
  tu.assert(colIndATA[5] == 0)
  tu.assert(colIndATA[6] == 2)
  tu.assert(colIndATA[7] == 3)
  tu.assert(colIndATA[8] == 0)
  tu.assert(colIndATA[9] == 1)
  tu.assert(colIndATA[10] == 2)
  tu.assert(colIndATA[11] == 3)

  tu.assert(valATA[0] == 50)
  tu.assert(valATA[1] == 59)
  tu.assert(valATA[2] == 4)
  tu.assert(valATA[3] == 81)
  tu.assert(valATA[4] == 18)
  tu.assert(valATA[5] == 59)
  tu.assert(valATA[6] == 73)
  tu.assert(valATA[7] == 12)
  tu.assert(valATA[8] == 4)
  tu.assert(valATA[9] == 18)
  tu.assert(valATA[10] == 12)
  tu.assert(valATA[11] == 20)
 
 C.free(rowPtrA)
 C.free(colIndA)
 C.free(valA)

 C.free(rowPtrAT)
 C.free(colIndAT)
 C.free(valAT)

 C.free(rowPtrATA)
 C.free(colIndATA)
 C.free(valATA)
end
computeNnzPatternAT_test()
