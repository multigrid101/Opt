from Test import ExampleRun
import re
import os
import subprocess as sp
import pdb
import sys

# Runs all examples with a single backend and many iterations to see if
# the outputs look reasonable. That way, we can detect bugs in the cpp-code
# and bugs that only occur in Opt  when using many solver iterations.
#
# Example-outputs need to be inspected by hand, but at least we can can catch
# run-time/compile-time problems.
# USAGE:
# > python test_example_output.py <backend>
#
# EXAMPLE:
# > python test_example_output.py backend_cpu
#
# default backend is cuda

# -----------------------------------------------------------------------------
# define list of folders to run:
folders = []

folders.append("arap_mesh_deformation")

folders.append("cotangent_mesh_smoothing")

folders.append("embedded_mesh_deformation")

folders.append("image_warping")

folders.append("intrinsic_image_decomposition")

folders.append("optical_flow")

folders.append("poisson_image_editing")

# # this is broken at the moment so ignore the error, need to find out when this happened
# # Run anyway to make sure we don't get compile-time or run-time errors
folders.append("robust_nonrigid_alignment") # note: this solver does more than one solve because singleSolve is overwritten by CombinedSolver

folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")
# -----------------------------------------------------------------------------



if len(sys.argv)>1:
    backends = [sys.argv[1]]
else:
    backends = ["backend_cuda"]



#------------------------------------------------------------------------------
# DEFINE ITERATION ARGS
iterArgs = {}
iterArgs['arap_mesh_deformation'] =         ["--oIterations", "10", "--nIterations", "20", "--lIterations", "100"]
iterArgs['cotangent_mesh_smoothing'] =      ["--oIterations", "1", "--nIterations", "5", "--lIterations", "25"] # oIterations is ignored
iterArgs['embedded_mesh_deformation'] =     ["--oIterations", "31", "--nIterations", "5", "--lIterations", "125"]
iterArgs['image_warping'] =                 ["--oIterations", "19", "--nIterations", "8", "--lIterations", "400"]
iterArgs['intrinsic_image_decomposition'] = ["--oIterations", "1", "--nIterations", "7", "--lIterations", "10"] # oIterations is ignored
iterArgs['optical_flow'] =                  ["--oIterations", "3", "--nIterations", "1", "--lIterations", "50"]
iterArgs['poisson_image_editing'] =         ["--oIterations", "1", "--nIterations", "1", "--lIterations", "100"] #oIterations is ignored

# TODO adjust to correct values once the example has been fixed.
iterArgs['robust_nonrigid_alignment'] =     ["--oIterations", "3", "--nIterations", "3", "--lIterations", "3"]

iterArgs['shape_from_shading'] =            ["--oIterations", "1", "--nIterations", "60", "--lIterations", "10"] # oIterations is ignored
iterArgs['volumetric_mesh_deformation'] =   ["--oIterations", "1", "--nIterations", "20", "--lIterations", "60"] # oIterations is ignored
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# DEFINE COMMAND FOR VIEWING OUTPUT
viewCommand = {}
viewCommand['arap_mesh_deformation'] =         ["meshlab", "out.ply"]
viewCommand['cotangent_mesh_smoothing'] =      ["meshlab", "out.off"]
viewCommand['embedded_mesh_deformation'] =     ["meshlab", "out.off"]
viewCommand['image_warping'] =                 ["feh", "output.png"]
viewCommand['intrinsic_image_decomposition'] = ["feh", "outputAlbedo.png", "outputShading.png"]
viewCommand['optical_flow'] =                  ["feh", "out.png"]
viewCommand['poisson_image_editing'] =         ["feh", "output.png"]

# TODO adjust to correct values once the example has been fixed.
viewCommand['robust_nonrigid_alignment'] =     "ls"

viewCommand['shape_from_shading'] =            ["meshlab", "sfsOutput.ply"]
viewCommand['volumetric_mesh_deformation'] =   ["meshlab", "out.ply"]
#------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Reference Costs from github.com/niessner/Opt cuda backend. Actual results
# may differ by as much as 20%, but that is to be expected due to the large
# problem sizes and large number of iterations. Still, we can at least detect
# if the results are way off.
#
# All costs refer to the FIRST outer iteration due to parsing reasons. Note
# that we always need to do the full amount of outer iterations (=n) in the
# reference examples and THEN take the value of the first cost (instead of just
# doing 1 outer iteration) or else the
# constraints (e.g. in image_warping) will be different in reference and here.
# Constraints are always moved 1/n-th of the final contraint value.
#
# We need to take the cost of the first outer iteration because it can
# be more easily parsed by regex (see below)
referenceCosts = {}
referenceCosts['arap_mesh_deformation'] = 3.929534
referenceCosts['cotangent_mesh_smoothing'] = 1712.155151 # CUDA
referenceCosts['embedded_mesh_deformation'] = 0.0944651290 # CUDA
referenceCosts['image_warping'] = 0.078505
referenceCosts['intrinsic_image_decomposition'] = 26004924
referenceCosts['optical_flow'] = 4055.0903320312
referenceCosts['poisson_image_editing'] = 139582.609375
referenceCosts['robust_nonrigid_alignment'] = -1 # FIXME
referenceCosts['shape_from_shading'] = 58.968529
referenceCosts['volumetric_mesh_deformation'] = 19.287743 # CUDA
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# tolerances for the various examples. The tolerances were obtained as follows:
# The original examples from github.com/niessner/Opt were run several times
# (10-100, depending on the problem size) with the cuda backend. We then recorded
# the largest deviations in those runs and define them as acceptable here.
# (actual tolerance may be slightly larger to allow for a little more error)
#
# all tolerances are *relative*
tols = {}
tols['arap_mesh_deformation'] = 1e-5
tols['cotangent_mesh_smoothing'] = 1e-3 # CUDA
tols['embedded_mesh_deformation'] = 1e-5 # CUDA
tols['image_warping'] = 1e-4 # CUDA
tols['intrinsic_image_decomposition'] = 1e-7
tols['optical_flow'] = 1e-5 # CUDA, result of first!!! level
tols['poisson_image_editing'] = 1e-7
tols['robust_nonrigid_alignment'] = -1 # FIXME
tols['shape_from_shading'] = 1e-2
tols['volumetric_mesh_deformation'] = 1e-6
# -----------------------------------------------------------------------------

#------------------------------------------------------------------------------
# FUNCTION TO PARSE OUTPUT
def getFinalCostFromRawOutput(output):
    match = re.search("final cost=(.*)", output)
    finalCost = float(match.group(1))
    return finalCost
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# RUN EVERYTHING
tests = []
for homedir in folders:
    print ""
    print 'begin ' + homedir
    for backend in backends:
        #-----------------------------------------------
        # create, define and run the example
        t = ExampleRun(homedir)
        t._printOutput = False
        t._execCommand = "./" + homedir
        t._args = ["--backend", backend,
            "--numthreads", "4"]
        t._args += iterArgs[homedir]
        # t._args += ["--useMaterializedJTJ", "--useFusedJTJ"]

        t.run()

        # get final cost from raw text output
        finalCost = float(getFinalCostFromRawOutput(t._output))
        print(finalCost)
        print(referenceCosts[homedir])
        #-----------------------------------------------

    print 'end ' + homedir
    print ""


# VIEW THE OUTPUTS
# for homedir in folders:
#         # enter directory
#         os.chdir(homedir)

#         # open output
#         callcommand = viewCommand[homedir]
#         print(callcommand)
#         print(os.listdir('.'))
#         process = sp.Popen(callcommand, stdout=sp.PIPE)

#         output = process.communicate()[0]
#         print(output)

#         # leave directory
#         os.chdir("..")

