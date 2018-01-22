local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
]]
local I = require('ittnotify')
local la = {} -- this module

-- IntList  helper needed below START
-- TODO adjust comments
local struct IntList {
  vals : &int -- array of values
  length : int -- current number of elements
  capacity : int -- maximum number of elements
}

terra IntList:init()
  -- 1000 should be enough for most purposes
  -- TODO free memory
  self.vals = [&int](C.malloc( 1000*sizeof(int) )) 
  C.memset([&opaque](self.vals), -1, 1000*sizeof(int))
  self.length = 0
  self. capacity = 1000
end

terra IntList:reset()
  self.length = 0
end

terra IntList:getLength()
  return self.length
end

terra IntList:append(a : int)
  -- TODO check capacity and resize if necessary
  if self.length+1 >= self.capacity then
    C.printf('\n\nERROR: IntList:append(): capacity exceeded, need to refactor code\n\n')
  end

  self.vals[self.length] = a
  self.length = self.length + 1
end

    -- required as comparison function for qsort() below
    local terra compare_ints(aptr : &opaque, bptr : &opaque) : int
      var a = @([&int](aptr))
      var b = @([&int](bptr))
      if a > b then
        return 1
      elseif a < b then
        return -1
      else
        return 0
      end
    end
terra IntList:sortInPlace()
  -- Note: Petsc uses
  --        - bubblesort if lenght<8
  --        - quicksort from K&R, page 87
  --        - See PetscSortInt() in sorti.c of petsc source
  C.qsort(self.vals, self.length, sizeof(int), compare_ints)
end

terra IntList:copyInto( v : &int )
-- copies values of self into some integer array
  for k = 0,self.length do
    v[k] = self.vals[k]
  end
end

terra IntList:getElement( idx : int )
-- copies values of self into some integer array
  if idx >= self.length then
    C.printf('\n\nERROR: IntList:getElement(): idx > length\n\n')
  elseif idx < 0 then
    C.printf('\n\nERROR: IntList:getElement(): idx < 0\n\n')
  end
  
  return self.vals[idx]
end
-- IntList  helper END

-- linalg stuff START
-- TODO doesn't work with double precision
-- TODO make sure that these functions work with opt_float
local terra getEntry(ri : int, ci : int, rowPtrA : &int, colIndA : &int, valA : &float) : float
-- finds entry in a csr matrix
-- First look through nnz entries of A. If A doesn't have an entry at the required
-- position, zero is returned.
-- No size-checks are performed!!!
  var offset = rowPtrA[ri]
  var nnzThisRow = rowPtrA[ri+1] - rowPtrA[ri]
  for k = 0,nnzThisRow do
    if ci == colIndA[offset+k] then
      return valA[offset+k]
    end
  end

  return 0
end

local terra addEntry(ri : int, ci : int,
                     rowPtrA : &int, colIndA : &int, valA : &float,
                     val : float)
-- does A[ri,ci] += val
-- First look through nnz entries of A. If A doesn't have an entry at the required
-- position, nothing is done
-- No size-checks are performed!!!
  var offset = rowPtrA[ri]
  var nnzThisRow = rowPtrA[ri+1] - rowPtrA[ri]

  for k = 0,nnzThisRow do
    if ci == colIndA[offset+k] then
      valA[offset+k] = valA[offset+k] + val
    end
  end
end

-- TODO create functions for sanity-checking of a matrix. (make sure that
-- colinds are sorted etc.

-- TODO create Matrix-class (similar to petsc)

-- TODO move these print* functions to a linalg_helper file because we mos
-- likely need them in all cpu backends (at least for testing)
local terra printValuesA(nrows : int, ncols : int, nnz : int,
                             rowPtrA : &int, colIndA : &int, valA : &float)
  C.printf('----------  MAT START ----------\n')
  C.printf('\n')
  for row = 0,nrows do -- print a row
    var nnzThisRow = rowPtrA[row+1] - rowPtrA[row]
    var currentCol = 0
    for k = 0, nnzThisRow do
    -- print all nnz values in the row
      var ci = colIndA[rowPtrA[row] + k]
      if currentCol == ci then
        -- we are printing the currentCol-th screen column. If the next colInd
        -- (ci) of the matrix is equal to currentCol, then we print
        -- the corresponding value......or......
        C.printf('%.1f ', valA[rowPtrA[row] + k])
        currentCol = currentCol + 1
      else
        -- ....otherwise, we print zero-screen-columns until currentCol has reached
        -- the next colInd in the matrix.....
        for k = 0,(ci-currentCol) do
          C.printf('%.1f ', 0.0f)
          currentCol = currentCol + 1
        end
        -- and then print the value corresponding to colInd and advance to the
        -- next loop iteration (i.e. next colInd)
        C.printf('%.1f ', valA[rowPtrA[row] + k])
        currentCol = currentCol + 1
      end
    end
    -- If the highest colInd is smaller than the number of columns in the
    -- matrix, we need to fill the end of the row with zeroes. E.g. if the
    -- colInds for the current row are (0 4 5), but the matrix is 8x10, then
    -- we need to append 4 zeroes (for indices 6,7,8,9) to the end of the row
    -- after printing the value for colInd=5.
    for k = 0,(ncols-currentCol) do
      C.printf('%.1f ', 0.0f)
    end
    -- finish row with newline and empty row before advancing to next row
    C.printf('\n\n')
  end -- finished printing the row
  C.printf('----------  MAT END ----------\n')
end
la.printValuesA = printValuesA

local terra printNnzPatternA(nrows : int, ncols : int, nnz : int,
                             rowPtrA : &int, colIndA : &int)
-- see other print* function for explanation of the code.
  C.printf('----------  MAT START ----------\n')
  C.printf('\n')
  for row = 0,nrows do
    var nnzThisRow = rowPtrA[row+1] - rowPtrA[row]
    var currentCol = 0
    for k = 0, nnzThisRow do
      var ci = colIndA[rowPtrA[row] + k]
      if currentCol == ci then
        C.printf('* ', ci)
        currentCol = currentCol + 1
      else
        for k = 0,(ci-currentCol) do
          C.printf('  ')
          currentCol = currentCol + 1
        end
        C.printf('* ')
        currentCol = currentCol + 1
      end
    end
    for k = 0,(ncols-currentCol) do
      C.printf('  ')
    end
    C.printf('\n\n')
  end
  C.printf('----------  MAT END ----------\n')
end
la.printNnzPatternA = printNnzPatternA


local terra computeNnzPatternAT(handle : &opaque, -- needed by cusparse lib TODO refactor
                                descr : &opaque, -- needed by cusparse lib TODO refactor
                                nColsA : int, -- if A is nxm, then this is m
                                nRowsA : int, -- if A is nxm, then this is n
                                nnzA : int,
                                rowPtrA : &int, colIndA : &int,
                                rowPtrAT : &int, colIndAT : &int)
-- TODO change interface to have nRows arg before nCols arg.
-- output: rowPtrAT, colIndAT. They must be allocated BEFORE this function is
-- called.

  --    Note: This section is mostly based on the implementation of a matrix
  --    transpose in Petsc, see 'MatGetSymbolicTranspose_SeqAIJ()' in symtranspose.c
  --    of their source code.
  var nRowsATA = nColsA
  var nColsATA = nColsA
  var nRowsAT = nColsA
  var nnzAT = nnzA
  var nnzATA = 0

  var name_malloc = I.__itt_string_handle_create("malloc/memset")
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_malloc)

  var numNnzEntriesPerRowAT = [&int](C.malloc( nRowsAT*sizeof(int) ))

  C.memset([&opaque](numNnzEntriesPerRowAT), 0, nRowsAT*sizeof(int))
  C.memset([&opaque](rowPtrAT), 0, (nRowsAT+1)*sizeof(int))

  -- we need a helper array that tells us the next free space for each row
  -- in colIndAT (needed in section c)
  var nextFreeSpace = [&int](C.malloc( nnzAT*sizeof(int) ))
  C.memset([&opaque](nextFreeSpace), 0, nRowsAT*sizeof(int))

  I.__itt_task_end(domain)


  -- a) traverse A and if A[i,j] != 0, we increment numNnzEntriesPerRowAT[j],
  -- i.e. count nnz per row of AT
  var name_count = I.__itt_string_handle_create("count nnz")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_count)
  for riA = 0,nRowsA do
    var nnzThisRowA = rowPtrA[riA+1] - rowPtrA[riA]
    var ptrTo_ThisRowInColIndA = &(colIndA[rowPtrA[riA]])
    for k = 0,nnzThisRowA do
      var ciA = ptrTo_ThisRowInColIndA[k]
      var riAT = ciA
      numNnzEntriesPerRowAT[riAT] = numNnzEntriesPerRowAT[riAT] + 1
    end
  end
  I.__itt_task_end(domain)

  -- b) Now we know how many entries each row in AT has, we use this to construct
  --     rowPtrAT
  rowPtrAT[0] = 0
  for riAT = 0,nRowsAT do
    var nextOffset : int -- offset of next row
    var thisOffset : int -- offset of current row
    var numNnzThisRow : int = numNnzEntriesPerRowAT[riAT]
    thisOffset = rowPtrAT[riAT]
    nextOffset = thisOffset + numNnzThisRow
    rowPtrAT[riAT+1] = nextOffset
  end

  -- c) We do a second traversal through A and fill in colIndAT by using
  -- the offsets in rowPtrAT
  -- As a nice side-effect, the traversal through A also ensures that the colinds
  -- of AT are sorted for each row, because we first insert riA=0 into the colinds
  -- of AT whereever they are present, then riA=1 and so on...
  var name_colind = I.__itt_string_handle_create("colIndAT")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_colind)
  for riA = 0,nRowsA do
    var nnzThisRowA = rowPtrA[riA+1] - rowPtrA[riA]
    var ptrTo_ThisRowInColIndA = &(colIndA[rowPtrA[riA]])
    for k = 0,nnzThisRowA do
      var ciA = ptrTo_ThisRowInColIndA[k]
      var riAT = ciA
      colIndAT[rowPtrAT[riAT] + nextFreeSpace[riAT]] = riA
      nextFreeSpace[riAT] = nextFreeSpace[riAT] + 1
    end
  end
  I.__itt_task_end(domain)

  C.free( numNnzEntriesPerRowAT )
  C.free( nextFreeSpace )
end
la.computeNnzPatternAT = computeNnzPatternAT


local terra computeNnzPatternATA(handle : &opaque, -- needed by cusparse lib TODO refactor
                                descr : &opaque, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                rowPtrA : &int, colIndA : &int,
                                rowPtrATA : &int, ptrTo_colIndATA : &&int, nnzATAptr : &int)
-- This function computes rowPtrATA AND colPtrATA. Both are used by computeATA()
-- to compute the values in AT*A.
-- Note: To compute the nnz pattern of ATA, we first compute the nnz pattern of
--       AT in one step and then the nnz pattern of ATA in a second step. This
--       two-step approach is based on the Petsc source code.
--       (MatTransposeMatMult_Symbolic_SeqAIJ_SeqAIJ() in matmatmult.c)

  -- 1) First compute nnz pattern of AT as a temporary helper
  --    Note: This section is mostly based on the implementation of a matrix
  --    transpose in Petsc, see 'MatGetSymbolicTranspose_SeqAIJ()' in symtranspose.c
  --    of their source code.
  var nRowsATA = nUnknowns
  var nColsATA = nUnknowns
  var nRowsAT = nUnknowns
  var nRowsA = nResiduals
  var nColsA = nUnknowns
  var nnzAT = nnzA
  var nnzATA = 0

  var rowPtrAT = [&int](C.malloc( (nRowsAT+1)*sizeof(int) ))
  var colIndAT = [&int](C.malloc( nnzAT*sizeof(int) ))

  var name_AT_helper = I.__itt_string_handle_create("AT helper")
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_AT_helper)

  computeNnzPatternAT(nil, nil, nColsA, nRowsA, nnzA,
                      rowPtrA, colIndA,
                      rowPtrAT, colIndAT)

  I.__itt_task_end(domain)
  


  -- 2) Now we have both AT and A in csr format, so we can compute the nnz
  -- pattern of ATA.
  -- Note: this section is also based on Petsc source, see
  -- MatMatMult_Symbolic_SeqAIJ_SeqAIJ() in matmatmult.c

   -- helper array: We set it to zero and for each row in ATA use it to keep
   -- track of which column inds we have already seen while merging the colinds
   -- of rows of A.

   var seen = [&bool](C.malloc( nRowsATA*sizeof(uint8) ))
   C.memset([&opaque](seen), 0, nRowsATA*sizeof(bool))
   var uniquelist : IntList
   uniquelist:init()

  -- 2a) first traversal to compute rowPtrATA and get nnzATA
  var name_rowPtrATA = I.__itt_string_handle_create("rowPtrATA")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_rowPtrATA)

  rowPtrATA[0] = 0
  for riATA = 0,nRowsATA do -- compute a row of ATA
    var riAT = riATA
    var ptrTo_ThisRowInColIndAT = &(colIndAT[rowPtrAT[riAT]]) -- colinds in this row in AT
    var nnzThisRowAT = rowPtrAT[riAT+1] - rowPtrAT[riAT]

    var nnzThisRowATA = 0
    for k = 0,nnzThisRowAT do --iterate through rows in A and merge their colInds
      var ciAT = ptrTo_ThisRowInColIndAT[k]
      var riA = ciAT
      var ptr_toRowInColIndA = &(colIndA[rowPtrA[riA]])
      var nnzInRowA = rowPtrA[riA+1] - rowPtrA[riA]

      for j = 0,nnzInRowA do
        var ciATA = ptr_toRowInColIndA[j]
        -- insert columnindex into list of unique colinds, but only if it hasn't
        -- been seen yet for the current row of ATA
        if (not seen[ciATA]) then
          nnzATA = nnzATA + 1
          nnzThisRowATA = nnzThisRowATA + 1
          seen[ciATA] = true
          uniquelist:append(ciATA)
        end
      end
    end -- finished merging colinds of rows of A

    -- set rowPtrATA for next row
    rowPtrATA[riATA+1] = rowPtrATA[riATA] + nnzThisRowATA

    -- reset 'seen' array and list of column indices for current row
    -- for next iteration
    for k = 0,nnzThisRowATA do seen[uniquelist:getElement(k)] = false end
    uniquelist:reset()
  end

  I.__itt_task_end(domain)

  -- 2b) allocate colIndATA
  -- Note that this memory isn't freed here because it's actually a return arg
  -- of this function
  var colIndATA = [&int](C.malloc( nnzATA * sizeof(int) ))
  C.memset([&opaque](colIndATA), -1, nnzATA * sizeof(int))

  -- 2c) second traversal to compute colIndATA
  var name_colIndATA = I.__itt_string_handle_create("colIndATA")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_colIndATA)

  for riATA = 0,nRowsATA do -- compute a row of ATA
    var riAT = riATA
    var ptrTo_ThisRowInColIndAT = &(colIndAT[rowPtrAT[riAT]]) -- colinds in this row in AT
    var nnzThisRowAT = rowPtrAT[riAT+1] - rowPtrAT[riAT]

    for k = 0,nnzThisRowAT do --iterate through rows in A and merge their colInds
      var ciAT = ptrTo_ThisRowInColIndAT[k]
      var riA = ciAT
      var ptr_toRowInColIndA = &(colIndA[rowPtrA[riA]])
      var nnzInRowA = rowPtrA[riA+1] - rowPtrA[riA]

      for j = 0,nnzInRowA do
        var ciATA = ptr_toRowInColIndA[j]
        -- insert columnindex into list of unique colinds, but only if it hasn't
        -- been seen yet for the current row of ATA
        if (not seen[ciATA]) then
          seen[ciATA] = true
          uniquelist:append(ciATA)
        end
      end
    end -- finished merging colinds of rows of A

    -- sort colinds
    uniquelist:sortInPlace()

    -- get nnz for current row of ATA
    var nnzThisRowATA = rowPtrATA[riATA+1] - rowPtrATA[riATA]

    -- get offset into colind array for current row of ATA
    var ptrTo_ThisRowInColIndATA = &(colIndATA[rowPtrATA[riATA]])

    -- copy colinds from list into colIndATA array
    uniquelist:copyInto( ptrTo_ThisRowInColIndATA )

    -- reset 'seen' array and list of column indices for current row
    -- for next iteration
    for k = 0,nnzThisRowATA do seen[uniquelist:getElement(k)] = false end
    uniquelist:reset()
  end

  I.__itt_task_end(domain)
  
  @ptrTo_colIndATA = colIndATA
  @nnzATAptr = nnzATA

  C.free( rowPtrAT )
  C.free( colIndAT )
  C.free( seen )
end
la.computeNnzPatternATA = computeNnzPatternATA


local terra computeATA(handle : &opaque, -- needed by cusparse lib TODO refactor
                                descr : &opaque, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                nnzATA : int,
                                valA : &float, rowPtrA : &int, colIndA : &int,
                                valAT : &float, rowPtrAT : &int, colIndAT : &int,
                                valATA : &float, rowPtrATA : &int, colIndATA : &int) -- valATA(out)
-- computes only the values of ATA, everything else needs to be provided.
-- Note that this is slightly different from
-- the implementation in the cuda backend, which also computes colIndATA in this
-- function. This is also why the name for this and other function in this 
-- file may be slightly confusing.
-- Also, colIndA needs to be sorted

-- This function is based on the Petsc source code
-- (MatMatMultNumeric_SeqAIJ_SeqAIJ[_Scalable]() in matmatmult.c)
-- TODO compare the performance of the two different petsc versions mentioned above
-- TODO investigate possible optimizations due to symmetry of ATA

  var nRowsATA = nUnknowns

  var name_zeroing = I.__itt_string_handle_create("memset 0")
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_zeroing)

  C.memset([&opaque](valATA), 0, nnzATA * sizeof(float))

  I.__itt_task_end(domain)


  var name_compute = I.__itt_string_handle_create("compute values")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_compute)
  
  for riATA = 0,nRowsATA do -- compute each row of ATA separately
    -- C.printf('doing row %d of ATA:\n', riATA)
    var riAT = riATA
    var offsetThisRowAT = rowPtrAT[riATA]
    var offsetThisRowATA = rowPtrATA[riATA]
    var nnzThisRowAT = rowPtrAT[riAT+1] - rowPtrAT[riAT]
    var nnzThisRowATA = rowPtrATA[riATA+1] - rowPtrATA[riATA]

    -- loop through all nnz entries in current row of AT
    for k = 0,nnzThisRowAT do
      var tmpval = valAT[offsetThisRowAT + k]
      var ciAT = colIndAT[offsetThisRowAT + k]
      var offsetThisRowA = rowPtrA[ciAT]
      var ciA = 0
      var nnzThisRowA = rowPtrA[ciAT+1] - rowPtrA[ciAT]
      var nnzFoundInA = 0

      -- do sparse (TODO try dense) axpy for each row in A that corresponds to
      -- a nnz in th current row of AT
      for l = 0,nnzThisRowATA do
        if nnzFoundInA < nnzThisRowA then -- check to avoid segfault in next line
          -- C.printf('offsetThisRATA: %d/%d and offsetThisRowA %d/%d\n', offsetThisRowATA, l, offsetThisRowA,ciA)
          if colIndATA[offsetThisRowATA + l] == colIndA[offsetThisRowA + ciA] then
            valATA[offsetThisRowATA + l] = valATA[offsetThisRowATA + l] 
                                         + tmpval*valA[offsetThisRowA + ciA]
            ciA = ciA + 1
            nnzFoundInA = nnzFoundInA + 1
          end
        end
      end
    end
  end

  I.__itt_task_end(domain)
end
la.computeATA = computeATA


local terra computeAT(handle : &opaque, -- needed by cusparse lib TODO refactor
                                descr : &opaque, -- needed by cusparse lib TODO refactor
                                nUnknowns : int, -- if A is nxm, then this is m
                                nResiduals : int, -- if A is nxm, then this is n
                                nnzA : int,
                                valA : &float, rowPtrA : &int, colIndA : &int,
                                valAT : &float, rowPtrAT : &int, colIndAT : &int) -- valATA(out), rowATA(int), colATA(out)
  -- computes the values of AT. Everything else (including rowPtrAT and colIndAT)
  -- have to be provided.
  -- colIndA needs to be sorted

  -- Assuming that nColsA ~ nRowsA, it doesn't really matter whether we iterate
  -- through A or AT (efficiency-wise). But since we need to *read* from
  -- A and *write* to AT, it is better to iterate through AT (this discussion
  -- isn't really relevant here but later for parallelization)
  -- TODO see if alternative implementation (iterate through A) is cheaper for
  -- single-threaded version.
  var numRowsAT = nUnknowns

  var name_zeroing = I.__itt_string_handle_create("memset 0")
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_zeroing)
  
  C.memset( [&opaque](valAT), 0, nnzA * sizeof(float) )

  I.__itt_task_end(domain)

  var name_compute = I.__itt_string_handle_create("compute values")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_compute)

  for riAT = 0,numRowsAT do
    var offsetThisRow = rowPtrAT[riAT]
    var nnzThisRowAT = rowPtrAT[riAT+1] - rowPtrAT[riAT]

    for k = 0,nnzThisRowAT do
      var ciAT = colIndAT[offsetThisRow + k]

      var valA = getEntry(ciAT, riAT, rowPtrA, colIndA, valA)

      valAT[offsetThisRow+k] = valA
    end
  end

  I.__itt_task_end(domain)
end
la.computeAT = computeAT


local terra applyAtoVector(handle : &opaque, -- needed by cusparse lib TODO refactor
                           descr : &opaque, -- needed by cusparse lib TODO refactor
                           nColsA : int, -- if A is nxm, then this is m
                           nRowsA : int, -- if A is nxm, then this is n
                           nnzA : int,
                           valA : &float, rowPtrA : &int, colIndA : &int,
                           valInVec : &float, valOutVec : &float,
                           bounds : &int) -- valInVec(in), valOutVec(out)
-- performs 'valOutVec <- A*valInVec'
-- bounds arg is ignored in this backend

  -- reset outVec
  var name_malloc = I.__itt_string_handle_create("malloc/memset")
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_malloc)

  C.memset([&opaque](valOutVec), 0, nRowsA * sizeof(float))

  I.__itt_task_end(domain)

-- TODO investigate optimizations due to symmetric A
        var start : C.timeval
        var stop : C.timeval
        var elapsed : double
        C.gettimeofday(&start, nil)

  var name_compute = I.__itt_string_handle_create("compute vals")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_compute)

  for k = 0,nRowsA do
    var offsetThisRowA = rowPtrA[k]
    var nnzThisRowA = rowPtrA[k+1] - rowPtrA[k]

    var tmp : float = 0.0f
    for l = 0,nnzThisRowA do
      tmp = tmp + valInVec[colIndA[offsetThisRowA+l]] * valA[offsetThisRowA+l]
    end
    valOutVec[k] = tmp
  end

  I.__itt_task_end(domain)

        C.gettimeofday(&stop, nil)
        elapsed = 1000*(stop.tv_sec - start.tv_sec)
        elapsed = elapsed + (stop.tv_usec - start.tv_usec)/(double)(1e3)
        C.printf("loop time was %f\n", elapsed)
end

-- optimized to exploit symmetry. Naive optimizations does not seems to work
-- In theory, we avoid loadint Aji if we already loaded Aij but in practice,
-- there are only few nnz per row so due to cachelines, we end up loading
-- everything anyway and furthermore have to do extra computations (if-stmts)
-- It seems that the matrix-format needs to be changed for symmetric matrices,
-- i.e. the extra values have to be "squeezed out".
local terra applyAtoVectorSym(handle : &opaque, -- needed by cusparse lib TODO refactor
                           descr : &opaque, -- needed by cusparse lib TODO refactor
                           nColsA : int, -- if A is nxm, then this is m
                           nRowsA : int, -- if A is nxm, then this is n
                           nnzA : int,
                           valA : &float, rowPtrA : &int, colIndA : &int,
                           valInVec : &float, valOutVec : &float,
                           bounds : &int) -- valInVec(in), valOutVec(out)
-- performs 'valOutVec <- A*valInVec'
-- bounds arg is ignored in this backend

  -- reset outVec
  var name_malloc = I.__itt_string_handle_create("malloc/memset")
  var domain = I.__itt_domain_create("Main.Domain")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_malloc)

  C.memset([&opaque](valOutVec), 0, nRowsA * sizeof(float))

  I.__itt_task_end(domain)

        var start : C.timeval
        var stop : C.timeval
        var elapsed : double
        C.gettimeofday(&start, nil)

  var name_compute = I.__itt_string_handle_create("compute vals")
  I.__itt_task_begin(domain, I.__itt_null, I.__itt_null, name_compute)

  for k = 0,nRowsA do
    var offsetThisRowA = rowPtrA[k]
    var nnzThisRowA = rowPtrA[k+1] - rowPtrA[k]

    var tmp : float = 0.0f
    for l = 0,nnzThisRowA do
      -- only update if we are in the upper-right half of the matrix (col>row)
      -- also update for lower-half via scatter.
      var colIndA = colIndA[offsetThisRowA+l]
      if colIndA > k then
        var valA = valA[offsetThisRowA+l]
        tmp = tmp + valInVec[colIndA] * valA
        valOutVec[colIndA] = valOutVec[colIndA] + valInVec[k]*valA
      else if colIndA == k then -- diagonal
        var valA = valA[offsetThisRowA+l]
        tmp = tmp + valInVec[colIndA] * valA
      end
      end
    end
    valOutVec[k] = tmp
  end

  I.__itt_task_end(domain)

        C.gettimeofday(&stop, nil)
        elapsed = 1000*(stop.tv_sec - start.tv_sec)
        elapsed = elapsed + (stop.tv_usec - start.tv_usec)/(double)(1e3)
        C.printf("loop time was %f\n", elapsed)
end
la.applyAtoVector = applyAtoVector
-- la.applyAtoVector = applyAtoVectorSym


local terra initMatrixStuff(handlePtr : &opaque, descrPtr : &opaque)
-- this function needs to do some stuff in cuda backend, but not here.
end
la.initMatrixStuff = initMatrixStuff
-- linalg stuff END

-- this needs to do stuff in cpu_mt backend but is just a dummy here
local terra computeBoundsA(bounds : &int, rowPtrA : &int,
                           nnzA : int, numRowsA : int)
end
la.computeBoundsA = computeBoundsA

return la
