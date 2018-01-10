import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

from Test import ExampleRun
import outParse as prs
import myPlots as mp
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

    # get final cost from raw text output
    # print(t_opt._output)
    # print(t_cerLin._output)
    # PCGStep1AvgCost = float(prs.getNamedAverageTimeFromOutput('PCGStep1', t_opt._output))
    # PCGStep2AvgCost = float(prs.getNamedAverageTimeFromOutput('PCGStep2', t_opt._output))
    # PCGStep3AvgCost = float(prs.getNamedAverageTimeFromOutput('PCGStep3', t_opt._output))

    PCGInit1AvgCost = float(prs.getNamedAverageTimeFromOutput('PCGInit1', t_opt._output))
    PCGLinearUpdateAvgCost = float(prs.getNamedAverageTimeFromOutput('PCGLinearUpdate', t_opt._output))
    computeCostAvgCost = float(prs.getNamedAverageTimeFromOutput('computeCost', t_opt._output))

    # OptCostLinIter = PCGStep1AvgCost + PCGStep2AvgCost + PCGStep3AvgCost
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


# doTimingsCeresVsOptCpu('arap_mesh_deformation')


