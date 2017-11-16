local C = terralib.includecstring [[
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
]]
local la = {} -- this module

tp = require('threadpool')

c = require('config')
numthreads = c.numthreads
-- numthreads = 3


-- IntList  helper needed below START
-- TODO adjust comments
local struct IntList {
  vals : &int -- array of values
  length : int -- current number of elements
  capacity : int -- maximum number of elements
}

terra IntList:init()
  -- 1000 should be enough for most purposes
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
  -- TODO check capacity
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

local terra printValuesA(nrows : int, ncols : int, nnz : int,
                             rowPtrA : &int, colIndA : &int, valA : &float)
  C.printf('----------  MAT START ----------\n')
  C.printf('\n')
  for row = 0,nrows do
    var nnzThisRow = rowPtrA[row+1] - rowPtrA[row]
    var currentCol = 0
    for k = 0, nnzThisRow do
      var ci = colIndA[rowPtrA[row] + k]
      if currentCol == ci then
        C.printf('%.1f ', valA[rowPtrA[row] + k])
        currentCol = currentCol + 1
      else
        for k = 0,(ci-currentCol) do
          C.printf('%.1f ', 0.0f)
          currentCol = currentCol + 1
        end
        C.printf('%.1f ', valA[rowPtrA[row] + k])
        currentCol = currentCol + 1
      end
    end
    for k = 0,(ncols-currentCol) do
      C.printf('%.1f ', 0.0f)
    end
    C.printf('\n\n')
  end
  C.printf('----------  MAT END ----------\n')
end
la.printValuesA = printValuesA

local terra printNnzPatternA(nrows : int, ncols : int, nnz : int,
                             rowPtrA : &int, colIndA : &int)
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

  var numNnzEntriesPerRowAT = [&int](C.malloc( nRowsAT*sizeof(int) ))

  C.memset([&opaque](numNnzEntriesPerRowAT), 0, nRowsAT*sizeof(int))
  C.memset([&opaque](rowPtrAT), 0, (nRowsAT+1)*sizeof(int))

  -- a) traverse A and if A[i,j] != 0, we increment numNnzEntriesPerRowAT[j]
  for riA = 0,nRowsA do
    var nnzThisRowA = rowPtrA[riA+1] - rowPtrA[riA]
    var ptrTo_ThisRowInColIndA = &(colIndA[rowPtrA[riA]])
    for k = 0,nnzThisRowA do
      var ciA = ptrTo_ThisRowInColIndA[k]
      var riAT = ciA
      numNnzEntriesPerRowAT[riAT] = numNnzEntriesPerRowAT[riAT] + 1
    end
  end

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

  -- we need a helper array that tells us the next free space for each row
  -- in colIndAT
  var nextFreeSpace = [&int](C.malloc( nnzAT*sizeof(int) ))
  C.memset([&opaque](nextFreeSpace), 0, nRowsAT*sizeof(int))

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

  computeNnzPatternAT(nil, nil, nColsA, nRowsA, nnzA,
                      rowPtrA, colIndA,
                      rowPtrAT, colIndAT)
  


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

  -- 2b) allocate colIndATA
  -- Note that this memory isn't freed here because it's actually a return arg
  -- of this function
  var colIndATA = [&int](C.malloc( nnzATA * sizeof(int) ))
  C.memset([&opaque](colIndATA), -1, nnzATA * sizeof(int))

  -- 2c) second traversal to compute colIndATA
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

  C.memset([&opaque](valATA), 0, nnzATA * sizeof(float))
  
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
  -- A and *write* to AT, it is better to iterate through AT because that way
  -- we avoid race conditions and possibly expensive synchronization.
  -- TODO see if alternative implementation (iterate through A) is cheaper for
  -- single-threaded version.
  var numRowsAT = nUnknowns
  
  C.memset( [&opaque](valAT), 0, nnzA * sizeof(float) )

  for riAT = 0,numRowsAT do
    var offsetThisRow = rowPtrAT[riAT]
    var nnzThisRowAT = rowPtrAT[riAT+1] - rowPtrAT[riAT]

    for k = 0,nnzThisRowAT do
      var ciAT = colIndAT[offsetThisRow + k]

      var valA = getEntry(ciAT, riAT, rowPtrA, colIndA, valA)

      valAT[offsetThisRow+k] = valA
    end
  end
end
la.computeAT = computeAT


-- multi-threaded version START
-- TODO it should be possible (although difficult) to develop code-transformations
-- similar to openmp to transform the serial implementation of this function
-- into the parallel one
local function lowerbound(limitQuote, tid)
  return `[tid] * ([limitQuote] / [numthreads])
end
local function upperbound(limitQuote, tid)
  return `([tid]+1) * ([limitQuote] / [numthreads])
end
local terra applyAtoVectorMultiThread(handle : &opaque, -- needed by cusparse lib TODO refactor
                           descr : &opaque, -- needed by cusparse lib TODO refactor
                           nColsA : int, -- if A is nxm, then this is m
                           nRowsA : int, -- if A is nxm, then this is n
                           nnzA : int,
                           valA : &float, rowPtrA : &int, colIndA : &int,
                           valInVec : &float, valOutVec : &float) -- valInVec(in), valOutVec(out)
  escape
    -- 1) Define loop bounds
    local function lowerbound(limitQuote, tid)
      if tid <= numthreads-1 then
        return `[tid] * ([limitQuote] / [numthreads])
      else
        error('\n\nERROR: lowerbound(): tid higher than number of threads\n\n')
      end
    end
    local function upperbound(limitQuote, tid)
      if tid < numthreads-1 then
        return `([tid]+1) * ([limitQuote] / [numthreads])
      elseif tid == numthreads-1 then
        return `limitQuote
      else
        error('\n\nERROR: upperbound(): tid higher than number of threads\n\n')
      end
    end

    -- 2) Define tasks (a task is a tuple ('void f(void* arg)', void* arg))
    -- 2a) Define threadData (the struct that is passed to the worker-thread function)
    local struct WorkerThreadData {
      valOutVec : &float,
      valInVec : &float,
      valA : &float,
      rowPtrA : &int,
      colIndA : &int
      nRowsA : int
    }

    -- 2b) define threadfunc()'s, i.e. the functions that are executed by each
    --     worker-thread
    local taskfuncs = {}
    for tid = 0,numthreads-1 do
      taskfuncs[tid] = terra(arg : &opaque)
        -- unpack arg
        var data = [&WorkerThreadData](arg)
        var valOutVec = data.valOutVec
        var valInVec = data.valInVec
        var valA = data.valA
        var rowPtrA = data.rowPtrA
        var colIndA = data.colIndA
        var nRowsA = data.nRowsA

        for k = [ lowerbound(`nRowsA,tid) ], [ upperbound(`nRowsA,tid) ] do
          var offsetThisRowA = rowPtrA[k]
          var nnzThisRowA = rowPtrA[k+1] - rowPtrA[k]

          for l = 0,nnzThisRowA do
            valOutVec[k] = valOutVec[k]
                         + valInVec[colIndA[offsetThisRowA+l]] * valA[offsetThisRowA+l]
          end
        end

        -- barrier worker-side
        tp.theKernelFinishedByAllThreadsBarrier:signal()

        -- return to infinit loop with the message that more work will follow,
        -- i.e. that we wish to remain in the infinite loop
        var moreWorkWillCome = true
        return moreWorkWillCome
      end
      taskfuncs[tid]:setname('taskfunc_applyAtoVector_' .. tostring(tid))
      print(taskfuncs[tid])
    end

    -- 3) define the launcher-function that is run by the main-thread.
    -- NOTE: This quote contains the whole body of the actual 'MatVecMult' function,
    -- everything else is just metaprogramming
    emit quote

  -- reset outVec
  C.memset([&opaque](valOutVec), 0, nRowsA * sizeof(float))

      -- C.printf('bla1\n')
      -- a) ensure that all threads are alive
      -- tp.ThreadsAliveBarrier:wait()
      --> this is not done by GPULauncher, so we probably don't have to do it
      --  here

      -- a) prepare WorkerThreadData (i.e. load with stuff)
      var theData : WorkerThreadData
      theData.valOutVec = valOutVec
      theData.valInVec = valInVec
      theData.valA = valA
      theData.rowPtrA = rowPtrA
      theData.colIndA = colIndA
      theData.nRowsA = nRowsA

      -- b) define Tasks
      var tasks : tp.Task_t[ numthreads ]
      escape
        for tid = 0,numthreads-1 do
          emit quote
            tasks[tid].taskfunction = [ taskfuncs[tid] ]
            tasks[tid].arg = &theData
          end
        end
      end

      -- call initial lock on barrier
      tp.theKernelFinishedByAllThreadsBarrier:initialLock()


      -- c) put Tasks into taskQueue.
      for tid = 0,[numthreads] do
        tp.theTaskQueue:set(tid, tasks[tid])
      end

      -- d) barrier main-side
      tp.theKernelFinishedByAllThreadsBarrier:wait()
      tp.theKernelFinishedByAllThreadsBarrier:finalUnlock()
    end
  end
end
print(applyAtoVectorMultiThread)


local terra applyAtoVectorSerial(handle : &opaque, -- needed by cusparse lib TODO refactor
                           descr : &opaque, -- needed by cusparse lib TODO refactor
                           nColsA : int, -- if A is nxm, then this is m
                           nRowsA : int, -- if A is nxm, then this is n
                           nnzA : int,
                           valA : &float, rowPtrA : &int, colIndA : &int,
                           valInVec : &float, valOutVec : &float) -- valInVec(in), valOutVec(out)
-- performs 'valOutVec <- A*valInVec'

  -- reset outVec
  C.memset([&opaque](valOutVec), 0, nRowsA * sizeof(float))

  for k = 0,nRowsA do
    var offsetThisRowA = rowPtrA[k]
    var nnzThisRowA = rowPtrA[k+1] - rowPtrA[k]

    for l = 0,nnzThisRowA do
      valOutVec[k] = valOutVec[k]
                   + valInVec[colIndA[offsetThisRowA+l]] * valA[offsetThisRowA+l]
    end
  end
end
-- la.applyAtoVector = applyAtoVectorSerial
la.applyAtoVector = applyAtoVectorMultiThread


local terra initMatrixStuff(handlePtr : &opaque, descrPtr : &opaque)
-- this function needs to do some stuff in cuda backend, but not here.
end
la.initMatrixStuff = initMatrixStuff
-- linalg stuff END

return la
