from Test import ExampleRun
import re
import pdb

# Runs all examples to see if their costs are correct. For this purpose, we
# use very small problem sizes and a very small number of iterations (usually just 1).
# This way, we mainly test that **Opt** is doing what it is supposed to do but
# we might miss some errors in Opt that can occur if we do more than one iteration
# (not very likely) or (more likely) errors that occur in the cpp-code if more
# than one outer iteration is used (e.g. in optical flow).
# USAGE:
# > python test_final_cost.py

# -----------------------------------------------------------------------------
# define list of folders to run:
folders = []

# The problem size here is a few hundred unknowns. We cannot go smaller because
# that causes some segfault in the cpp code. May we can fix this at a later point.
folders.append("arap_mesh_deformation")

folders.append("cotangent_mesh_smoothing")

folders.append("embedded_mesh_deformation")

folders.append("image_warping")

# the error here can be around 1e-5 with stride=1, so we use stride=12
folders.append("intrinsic_image_decomposition")

# even with oIterations=1, we will get two solver outputs because of the
# 'numLevels' parameter in combinedSolver.h (numLevels=2)
folders.append("optical_flow")

# the error here will be slightly too large (~1.1e-6) for 4 threads,
# so we use stride = 4
folders.append("poisson_image_editing")

# # this is broken at the moment so ignore the error, need to find out when this happened
# # Run anyway to make sure we don't get compile-time or run-time errors
folders.append("robust_nonrigid_alignment") # note: this solver does more than one solve because singleSolve is overwritten by CombinedSolver

# currently NOT INCLUDED because the up/downscaling will be complicated
# folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# List of reference costs together with
# reference costs for small problem size and (nIter, nNonlinIter, nLinIter) = (1,1,1)
# reference cost should always be taken from CUDA!!!
referenceCosts = {}
referenceCosts['arap_mesh_deformation'] = 7183.464843 # CUDA
referenceCosts['cotangent_mesh_smoothing'] = 2091.86303 # CUDA
referenceCosts['embedded_mesh_deformation'] = 0.367129057645 # CUDA
referenceCosts['image_warping'] = 1774.3405 # CUDA
referenceCosts['intrinsic_image_decomposition'] = 3.3105300000e6 #CUDA, stride=12 (53x30 px)
referenceCosts['poisson_image_editing'] = 1530364.25 # CUDA, stride=4 (112x80 px)
referenceCosts['optical_flow'] = 0.52119255 # CUDA, result of first!!! level
referenceCosts['robust_nonrigid_alignment'] = 66.784683 # CUDA cost of first!!! solve
referenceCosts['shape_from_shading'] = -1
referenceCosts['volumetric_mesh_deformation'] = 189.74081 # CUDA
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# list of strides, if -1, then program will use default stride, i.e. 1 for
# pixelstuff and 0 subdivisions for graph stuff
strides = {}
strides['arap_mesh_deformation'] = -1
strides['cotangent_mesh_smoothing'] = -1
strides['embedded_mesh_deformation'] = -1
strides['image_warping'] = -1
strides['intrinsic_image_decomposition'] = 12
strides['optical_flow'] = 16
strides['poisson_image_editing'] = 4
strides['robust_nonrigid_alignment'] = -1
strides['shape_from_shading'] = -1
strides['volumetric_mesh_deformation'] = -1
# -----------------------------------------------------------------------------

backends = ["backend_cuda", "backend_cpu", "backend_cpu_mt"]
# backends = ["backend_cpu", "backend_cpu_mt"]
numthreads = ["1", "2", "4", "8"]
# numthreads = ["1", "2", "4", "8", "16", "32"]
# numthreads = ["4", "8"]
matargs = [["--useMaterializedJTJ", "false", "useFusedJTJ", "false"],
           ["--useMaterializedJTJ", "true", "useFusedJTJ", "false"],
           ["--useMaterializedJTJ", "true", "useFusedJTJ", "true"]]
            


#------------------------------------------------------------------------------
# FUNCTION TO PARSE OUTPUT
def getFinalCostFromRawOutput(output):
    match = re.search("final cost=(.*)", output)
    finalCost = float(match.group(1))
    return finalCost
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# TEST CLASS
class Test:
    def __init__(self):
        self._expected = 0.0
        self._actual = -1.0
        self._isOk = False
        self._infoString = ""
        self._hasFinished = False
        self._error = -1

    def compare(self):
        self._hasFinished = True
        self._error = abs(self._expected - self._actual)/self._expected
        # all examples should pass with tol=1e-6 except if mentioned otherwise
        # in the comments at the top.
        self._isOk = abs(self._error) < 1e-5

    def printInfo(self):
        if self._hasFinished:
            if self._isOk:
                print("{}OK{}, error is ".format('\033[92m','\033[0m') + str(self._error))
            else:
                print("{}NOT OK: {}".format('\033[91m','\033[0m') + self._infoString)
                print("        error is " + str(self._error))
        else:
            print("UNDEFINED")


#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# RUN EVERYTHING
tests = []
for homedir in folders:
    print ""
    print 'begin ' + homedir
    for backend in backends:
        for num in numthreads:
            for matarg  in matargs:
                # only do simulation if combination of parameters makes sense.
                if (backend == "backend_cuda" or backend == "backend_cpu") and int(num) > 1:
                    break

                #-----------------------------------------------
                # create, define and run the example
                t = ExampleRun(homedir)
                t._printOutput = False
                t._execCommand = "./" + homedir
                t._args = ["--backend", backend,
                    "--numthreads", num]

                # append stride flag if necessary
                stride = strides[homedir]
                if stride > 0:
                    t._args += ['--stride', str(stride)]

                # append flags to run with materializedJTJ, etc.
                t._args += matarg


                t.run()
                #-----------------------------------------------


                # get final cost from raw text output
                finalCost = float(getFinalCostFromRawOutput(t._output))
                print(finalCost)

                # compare against reference cost and throw error if necessary
                test = Test()
                test._expected = referenceCosts[homedir]
                test._actual = finalCost
                test._infoString = str(t.getCallCommand())
                test.compare()
                tests.append(test)

    print 'end ' + homedir
    print ""


# print summary
for t in tests:
    t.printInfo()



