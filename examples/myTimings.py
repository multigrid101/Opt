import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

from Test import ExampleRun
import outParse as prs
import myPlots as mp
import myInfos as info
import pdb

# make plots pretty
plt.style.use('ggplot')

folders = []
folders.append("arap_mesh_deformation")

folders.append("image_warping")

folders.append("poisson_image_editing")

folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")


# 'homedir' is a string with the example directory, e.g.
# 'image_warping'
# saves output in e.g. 'image_warping/timings/ceresVsOptCpu.timings'
# see below for definition of output format
def doTimingsCeresVsOptCpu(homedir):
    iterArgsOpt = {}
    iterArgsCeres = {}

    iterArgsOpt =                 ["--oIterations", "1", "--nIterations", "2", "--lIterations", "4"]

    # iterArgsCeresLin =                 ["--oIterations", "1", "--nIterations", "1", "--lIterations", "500", "--useCeres", "--useOpt", "false"]
    # iterArgsCeresNew =                 ["--oIterations", "1", "--nIterations", "100", "--lIterations", "2", "--useCeres", "--useOpt", "false"]
    iterArgsCeresLin =                 ["--oIterations", "1", "--nIterations", "1", "--lIterations", "50", "--useCeres", "--useOpt", "false"]
    iterArgsCeresNew =                 ["--oIterations", "1", "--nIterations", "10", "--lIterations", "2", "--useCeres", "--useOpt", "false"]



    tests = []

    # create, define the example runs
    t_opt = ExampleRun(homedir)
    t_opt._printOutput = False
    t_opt._execCommand = "./" + homedir
    t_opt._args = ["--backend", 'backend_cpu']
    t_opt._args += iterArgsOpt

    t_cerLin = ExampleRun(homedir)
    t_cerLin._printOutput = False
    t_cerLin._execCommand = "./" + homedir
    t_cerLin._args = ["--backend", 'backend_cpu']
    t_cerLin._args += iterArgsCeresLin

    t_cerNew = ExampleRun(homedir)
    t_cerNew._printOutput = False
    t_cerNew._execCommand = "./" + homedir
    t_cerNew._args = ["--backend", 'backend_cpu']
    t_cerNew._args += iterArgsCeresNew


    # run them
    t_opt.run()
    t_cerLin.run()
    t_cerNew.run()

    PCGInit1AvgCost = float(prs.getNamedAverageTimeFromOutput('PCGInit1', t_opt._output))
    PCGLinearUpdateAvgCost = float(prs.getNamedAverageTimeFromOutput('PCGLinearUpdate', t_opt._output))
    computeCostAvgCost = float(prs.getNamedAverageTimeFromOutput('computeCost', t_opt._output))

    OptCostLinIter = prs.getNamedAverageTimeFromOutput('linear iteration', t_opt._output)


    OptCostPerNewton = PCGInit1AvgCost + PCGLinearUpdateAvgCost + computeCostAvgCost

    CeresCostLinIter = prs.getAvgTimeLinIterCeres(t_cerLin._output)
    CeresCostPerNewton = prs.getAvgTimePerNewtonCeres(t_cerNew._output)

    costs = {
            "optPerLinIter" : OptCostLinIter,
            "optPerNewton" : OptCostPerNewton,
            "ceresPerLinIter" : CeresCostLinIter,
            "ceresPerNewton" : CeresCostPerNewton
            }

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/ceresVsOptCpu" + ".timing"
    pk.dump(costs, open(filename, "wb"))



# time opt for different example-sizes, for graph-stuff, this
# corresponds to numSubdivides, for pixelstuff it corresponds to the stride.
# materialization = [matfree|JTJ|fusedJTJ]
exp0002Sizes = {}
exp0002Sizes['arap_mesh_deformation'] = [1, 2, 3, 4]
exp0002Sizes['cotangent_mesh_smoothing'] = [1, 2, 3, 4]
exp0002Sizes['embedded_mesh_deformation'] = [1, 2, 3, 4]
exp0002Sizes['image_warping'] = [1, 2, 3, 4]
exp0002Sizes['intrinsic_image_decomposition'] = [1, 2, 3, 4]
exp0002Sizes['optical_flow'] = [1, 2, 3, 4]
exp0002Sizes['poisson_image_editing'] = [1, 2, 3, 4]
exp0002Sizes['robust_nonrigid_alignment'] = [1, 2, 3, 4]
exp0002Sizes['shape_from_shading'] = [1, 2, 3, 4]
exp0002Sizes['volumetric_mesh_deformation'] = [1, 2, 3, 4]
def doTimingsExp000234(homedir, materialization):
    sizes = exp0002Sizes[homedir]

    expNumber = -1 # 'init'
    if materialization == "matfree":
        expNumber = 2
    elif materialization == "JTJ":
        expNumber = 3
    elif materialization =="fusedJTJ":
        expNumber = 4
    else:
        errMsg = """
        doTimingsExp000234(): invalid materialization:

        {0}

        valid string-options are 'matfree', 'JTJ', 'fusedJTJ'
        """
        sys.exit(errMsg)


    tests = []

    # collect the size of the UnknownVector here
    # problemSizes = sizes
    problemSizes = []

    perSolveTimes = []
    perNewtonTimes = []
    perLinIterTimes = []

    for size in sizes:
        # create, define the example runs
        t_opt = ExampleRun(homedir)
        t_opt._printOutput = False
        t_opt._execCommand = "./" + homedir
        t_opt._args = ["--nIterations", "10", "--lIterations", "10"]

        if materialization == "matfree":
            pass # this is default
        elif materialization == "JTJ":
            t_opt._args.append("--useMaterializedJTJ")
        elif materialization =="fusedJTJ":
            t_opt._args.append("--useMaterializedJTJ")
            t_opt._args.append("--useFusedJTJ")
        else:
            pass # error checking is done at top of file


        t_opt._args += [info.strideFlags[homedir], str(size)]

        # run them
        t_opt.run()

        # get final cost from raw text output
        print(t_opt._output)

        optPerSolveTime = prs.getAvgCategoryTimeOpt(t_opt._output, 1, materialization, homedir)
        optPerNewtonTime = prs.getAvgCategoryTimeOpt(t_opt._output, 2, materialization, homedir)
        optPerLinIterTime = prs.getAvgCategoryTimeOpt(t_opt._output, 3, materialization, homedir)

        perSolveTimes.append(optPerSolveTime)
        perNewtonTimes.append(optPerNewtonTime)
        perLinIterTimes.append(optPerLinIterTime)

        unknownSize = prs.getUnknownSizeFromOutput(t_opt._output)
        problemSizes.append(unknownSize)



    # TODO properly parse the number of unknowns or whatever

    timingData = {
            "problemSizes" : problemSizes,
            "optPerSolveTimes" : perSolveTimes,
            "optPerNewtonTimes" : perNewtonTimes,
            "optPerLinIterTimes" : perLinIterTimes
            }
    print(timingData)

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/exp000{0}".format(expNumber) + ".timing"



    pk.dump(timingData, open(filename, "wb"))

# doTimingsExp0002('arap_mesh_deformation')
# doTimingsExp0002('image_warping')

