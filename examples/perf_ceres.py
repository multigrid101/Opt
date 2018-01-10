from Test import ExampleRun
import re


# -----------------------------------------------------------------------------
# define list of folders to run:
folders = []

# folders.append("arap_mesh_deformation")

# folders.append("cotangent_mesh_smoothing")

# folders.append("embedded_mesh_deformation")

folders.append("image_warping")

# folders.append("intrinsic_image_decomposition")

# folders.append("poisson_image_editing")

# # do not include, results are garbage and performance data most likely useless
# # folders.append("robust_nonrigid_alignment") # note: this solver does more than one solve because singleSolve is overwritten by CombinedSolver

# folders.append("volumetric_mesh_deformation")
# -----------------------------------------------------------------------------
otherFlags = ["--materializedJTJ", "false", "--useOpt", "false", "--useCeres"]



# -----------------------------------------------------------------------------
# the flags for pixelstuff and graphstuff are different, see ArgParser.h.
strideFlags = {}
strideFlags['arap_mesh_deformation'] = "--numSubdivides"
strideFlags['cotangent_mesh_smoothing'] = "--numSubdivides"
strideFlags['embedded_mesh_deformation'] = "--numSubdivides"
strideFlags['image_warping'] = "--stride"
strideFlags['intrinsic_image_decomposition'] = "--stride"
strideFlags['poisson_image_editing'] = "--stride"
strideFlags['robust_nonrigid_alignment'] = "--numSubdivides"
strideFlags['volumetric_mesh_deformation'] = "--numSubdivides"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# list of list of strides, we want to use different values for all examples
# TODO
strides = {}
strides['arap_mesh_deformation'] = [1,2]
strides['cotangent_mesh_smoothing'] = [0]
strides['embedded_mesh_deformation'] = [0]
strides['image_warping'] = [1, 2, 3]
strides['intrinsic_image_decomposition'] = [1]
strides['poisson_image_editing'] = [1]
strides['robust_nonrigid_alignment'] = [0]
strides['volumetric_mesh_deformation'] = [0]
# -----------------------------------------------------------------------------



#------------------------------------------------------------------------------
# FUNCTION TO PARSE OUTPUT
def getOverallTimeFromOutput(output):
    match = re.search(" overall\s+\|[^|]+\|([^|]*)ms", output)
    # overallTime = float(match.group(1))
    overallTime = float(match.group(1))
    return overallTime

def getNamedAbsoluteTimeFromOutput(name, output):
    match = re.search(name + "[^|]+\|[^|]+\|([^|]*)ms", output)
    # overallTime = float(match.group(1))
    overallTime = float(match.group(1))
    return overallTime

def getNamedAverageTimeFromOutput(name, output):
    match = re.search(name + "[^|]+\|[^|]+\|([^|]*)ms\|([^|]*)ms", output)
    # overallTime = float(match.group(1))
    overallTime = float(match.group(2))
    return overallTime
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# TEST CLASS
class Test:
    def __init__(self):
        # TODO add more timing info here and also methods to parse the info
        self._overallTime = -1.0
        self._infoString = ""
        self._hasFinished = False
        self._error = -1

    # def compare(self):
    #     self._hasFinished = True
    #     self._error = abs(self._expected - self._actual)/self._expected
    #     # all examples should pass with tol=1e-6 except if mentioned otherwise
    #     # in the comments at the top.
    #     self._isOk = self._error < 1e-6

    def printInfo(self):
        print("alle meine entchen")
        if self._hasFinished:
            if self._isOk:
                pass
            else:
                pass
        else:
            pass


#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# RUN EVERYTHING
tests = []
for homedir in folders:
    print ""
    print 'begin ' + homedir
    for stride in strides[homedir]:
        # only do simulation if combination of parameters makes sense.
        # TODO fill section or delete comment

        #-----------------------------------------------
        # create, define and run the example
        t = ExampleRun(homedir)

        t._printOutput = True
        # t._printOutput = False

        t._execCommand = "./" + homedir
        t._args = [strideFlags[homedir], str(stride)]

        t.run()
        #-----------------------------------------------


        # get overall time
        t_overall = getOverallTimeFromOutput(t._output)
        t_linearAvg = getNamedAverageTimeFromOutput("linear iteration", t._output)
        t_linearAbs = getNamedAbsoluteTimeFromOutput("linear iteration", t._output)
        print(t_overall)
        print(t_linearAvg)
        print(t_linearAbs)

        # insert into data-table/data-base
        # TODO

    print 'end ' + homedir
    print ""


# print summary
# TODO



