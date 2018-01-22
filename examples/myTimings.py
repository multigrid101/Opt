import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

from Test import ExampleRun
import outParse as prs
import myInfos as info
# import myPlots as mp
import myInfos as info
import pdb
import sys

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
exp0001Sizes = {}
exp0001Sizes['arap_mesh_deformation']         = 4
exp0001Sizes['cotangent_mesh_smoothing']      = 4
exp0001Sizes['embedded_mesh_deformation']     = 4
exp0001Sizes['image_warping']                 = 1
exp0001Sizes['intrinsic_image_decomposition'] = 1
exp0001Sizes['optical_flow']                  = 1
exp0001Sizes['poisson_image_editing']         = 1
exp0001Sizes['robust_nonrigid_alignment']     = 1
exp0001Sizes['shape_from_shading']            = 1
exp0001Sizes['volumetric_mesh_deformation']   = 2
def doTimingsCeresVsOptCpu(homedir):
    #TODO do timings only for whatever backend works best in opt.
    #TODO need to define arglists for each example and materialization-strategy and problem-size
    iterArgsOpt = {}
    iterArgsCeres = {}

    # iterArgsOpt =                 ["--oIterations", "1", "--nIterations", "2", "--lIterations", "40"]
    iterArgsOpt =                 ["--oIterations", "1", "--nIterations", "2", "--lIterations", "4"]

    iterArgsCeresLin =                 ["--oIterations", "1", "--nIterations", "1", "--lIterations", "50", "--useCeres", "--useOpt", "false"]
    iterArgsCeresNew =                 ["--oIterations", "1", "--nIterations", "10", "--lIterations", "2", "--useCeres", "--useOpt", "false"]
    # iterArgsCeresLin =                 ["--oIterations", "1", "--nIterations", "1", "--lIterations", "500", "--useCeres", "--useOpt", "false"]
    # iterArgsCeresNew =                 ["--oIterations", "1", "--nIterations", "100", "--lIterations", "2", "--useCeres", "--useOpt", "false"]



    tests = []

    # create, define the example runs
    t_opt = ExampleRun(homedir)
    t_opt._printOutput = False
    t_opt._execCommand = "./" + homedir
    t_opt._args = ["--backend", 'backend_cpu']
    t_opt._args = ["--useMaterializedJTJ", '--useFusedJTJ']
    t_opt._args += iterArgsOpt
    t_opt._args += [info.strideFlags[homedir], str(exp0001Sizes[homedir])]

    t_cerLin = ExampleRun(homedir)
    t_cerLin._printOutput = False
    t_cerLin._execCommand = "./" + homedir
    t_cerLin._args = ["--backend", 'backend_cpu']
    t_cerLin._args += iterArgsCeresLin
    t_cerLin._args += [info.strideFlags[homedir], str(exp0001Sizes[homedir])]

    t_cerNew = ExampleRun(homedir)
    t_cerNew._printOutput = False
    t_cerNew._execCommand = "./" + homedir
    t_cerNew._args = ["--backend", 'backend_cpu']
    t_cerNew._args += iterArgsCeresNew
    t_cerNew._args += [info.strideFlags[homedir], str(exp0001Sizes[homedir])]


    # run them
    t_opt.run()
    t_cerLin.run()
    t_cerNew.run()


    optPerSolveTime = prs.getAvgCategoryTimeOpt(t_opt._output, 1, 'fusedJTJ', homedir)
    OptCostPerNewton = prs.getAvgCategoryTimeOpt(t_opt._output, 2, 'fusedJTJ', homedir)
    OptCostLinIter = prs.getAvgCategoryTimeOpt(t_opt._output, 3, 'fusedJTJ', homedir)


    # TODO ceres per-solve time
    CeresCostLinIter = prs.getAvgTimeLinIterCeres(t_cerLin._output)
    CeresCostPerNewton = prs.getAvgTimePerNewtonCeres(t_cerNew._output)

    costs = {
            "optPerLinIter" : OptCostLinIter,
            "optPerNewton" : OptCostPerNewton,
            "optPerSolve" : optPerSolveTime,
            "ceresPerLinIter" : CeresCostLinIter,
            "ceresPerNewton" : CeresCostPerNewton
            }

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/ceresVsOptCpu" + ".timing"
    pk.dump(costs, open(filename, "wb"))


def doTimingsExp0028(homedir):
    #TODO do timings only for whatever backend works best in opt.
    #TODO need to define arglists for each example and materialization-strategy and problem-size
    iterArgsOpt = {}
    iterArgsCeres = {}

    iterArgs =                 ["--oIterations", "1", "--nIterations", "1", "--lIterations", "1"]
    sizes = [1,2,3,4]
    unSizes = {'matfree':[], 'JTJ':[], 'fusedJTJ':[]}
    pdSizes = {'matfree':[], 'JTJ':[], 'fusedJTJ':[]}

    tests = []

    for mater in ['matfree', 'JTJ', 'fusedJTJ']:
        for size in sizes:
            # create, define the example runs
            t_opt = ExampleRun(homedir)
            t_opt._printOutput = False
            t_opt._execCommand = "./" + homedir
            t_opt._args = ["--backend", 'backend_cpu']
            t_opt._args += iterArgsOpt
            t_opt._args += ["--noOutput"]
            t_opt._args += [info.strideFlags[homedir], str(size)]


            if mater == "matfree":
                pass # this is default
            elif mater == "JTJ":
                t_opt._args.append("--useMaterializedJTJ")
            elif mater =="fusedJTJ":
                t_opt._args.append("--useMaterializedJTJ")
                t_opt._args.append("--useFusedJTJ")
            else:
                pass # error checking is done at top of file


            # run them
            t_opt.run()

            pdSize  = prs.getPlanDataSizeFromOutput(t_opt._output)
            unSize  = prs.getUnknownSizeFromOutput(t_opt._output)

            unSizes[mater].append(unSize)
            pdSizes[mater].append(pdSize)


    print(unSizes)
    print(pdSizes)
    costs = {
            "unknownSizes" : unSizes,
            "planDataSizes" : pdSizes,
            }

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/exp0028" + ".timing"
    pk.dump(costs, open(filename, "wb"))


exp0029Sizes = {}
# TODO do for smaller sizes, too, cuda scales badly for those.
exp0029Sizes['arap_mesh_deformation'] = [1, 2, 3, 4, 5, 6] #>6 leads to OOME for cuda-fusedjtj
exp0029Sizes['cotangent_mesh_smoothing'] = [0, 1, 2, 3, 4]
exp0029Sizes['embedded_mesh_deformation'] = [0, 1, 2, 3] # >3 leads to out-of-memory error in cuda-backend(fusedjtj)
exp0029Sizes['image_warping'] = [6, 7, 8, 9, 10, 11, 12] # < 6 OOME for cuda-fusedjtj
exp0029Sizes['intrinsic_image_decomposition'] = [4,5,6,7,8,9,10] #<4 leads to OOME in cuda-fusedjtj
exp0029Sizes['optical_flow'] = [5, 6, 7,8,9] # <4 OOME in cuda-fusedjtj
exp0029Sizes['poisson_image_editing'] = [4, 5, 6, 7,8,9,10] # <4 OOME in cuda-fusedjtj
exp0029Sizes['robust_nonrigid_alignment'] = [1, 2, 3, 4]
exp0029Sizes['shape_from_shading'] = [1, 2, 3, 4]
exp0029Sizes['volumetric_mesh_deformation'] = [0, 1, 2, 3, 4] # >5 throws out-of-mem error in cuda-backend(fusedJTJ)

exp0029FileArgs = {}
exp0029FileArgs['arap_mesh_deformation'] =         []
exp0029FileArgs['cotangent_mesh_smoothing'] =      []
exp0029FileArgs['embedded_mesh_deformation'] =     []
exp0029FileArgs['image_warping'] =                 ['--file', "2"]
exp0029FileArgs['intrinsic_image_decomposition'] = ['--file', '2']
exp0029FileArgs['optical_flow'] =                  ['--file', '2']
exp0029FileArgs['poisson_image_editing'] =         ['--file', '2']
exp0029FileArgs['volumetric_mesh_deformation'] =   []
def doTimingsExp0029(homedir):
    iterArgs =                 ["--oIterations", "1", "--nIterations", "2", "--lIterations", "100"]
    sizes = exp0029Sizes[homedir]

    tests = []

    times_cuda_matfree = []
    times_cuda_fusedjtj = []
    times_mt_fusedjtj = []

    unSizes = []
    pdSizes = []

    for size in sizes:
        # create, define the example runs
        t_cuda_matfree = ExampleRun(homedir)
        t_cuda_matfree._printOutput = False
        t_cuda_matfree._execCommand = "./" + homedir
        t_cuda_matfree._args = ["--backend", 'backend_cuda']
        t_cuda_matfree._args += iterArgs
        t_cuda_matfree._args += ["--noOutput"]
        t_cuda_matfree._args += [info.strideFlags[homedir], str(size)]
        t_cuda_matfree._args += exp0029FileArgs[homedir]

        t_cuda_fusedjtj = ExampleRun(homedir)
        t_cuda_fusedjtj._printOutput = False
        t_cuda_fusedjtj._execCommand = "./" + homedir
        t_cuda_fusedjtj._args = ["--backend", 'backend_cuda']
        t_cuda_fusedjtj._args += iterArgs
        t_cuda_fusedjtj._args += ["--noOutput", "--useMaterializedJTJ", "--useFusedJTJ"]
        t_cuda_fusedjtj._args += [info.strideFlags[homedir], str(size)]
        t_cuda_fusedjtj._args += exp0029FileArgs[homedir]

        t_mt_fusedjtj = ExampleRun(homedir)
        t_mt_fusedjtj._printOutput = False
        t_mt_fusedjtj._execCommand = "./" + homedir
        t_mt_fusedjtj._args = ["--backend", 'backend_cpu_mt', '--numthreads', '4']
        t_mt_fusedjtj._args += iterArgs
        t_mt_fusedjtj._args += ["--noOutput", "--useMaterializedJTJ", "--useFusedJTJ"]
        t_mt_fusedjtj._args += [info.strideFlags[homedir], str(size)]
        t_mt_fusedjtj._args += exp0029FileArgs[homedir]


        # run them
        t_cuda_matfree.run()
        t_cuda_fusedjtj.run()
        t_mt_fusedjtj.run()

        lin_cuda_matfree = prs.getNamedAverageTimeFromOutput('linear iteration', t_cuda_matfree._output)
        lin_cuda_fusedjtj = prs.getNamedAverageTimeFromOutput('linear iteration', t_cuda_fusedjtj._output)
        lin_mt_fusedjtj = prs.getNamedAverageTimeFromOutput('linear iteration', t_mt_fusedjtj._output)
        unSize  = prs.getUnknownSizeFromOutput(t_cuda_matfree._output)
        pdSize  = prs.getPlanDataSizeFromOutput(t_mt_fusedjtj._output)

        times_cuda_matfree.append(lin_cuda_matfree)
        times_cuda_fusedjtj.append(lin_cuda_fusedjtj)
        times_mt_fusedjtj.append(lin_mt_fusedjtj)
        unSizes.append(unSize)
        pdSizes.append(pdSize)

    print(times_cuda_matfree)
    print(times_cuda_fusedjtj)
    print(times_mt_fusedjtj)
    costs = {
            "unknownSizes" : unSizes,
            "plandataSizes" : pdSizes,
            "cudaMatfreeTimes" : times_cuda_matfree,
            "cudaFusedJTJTimes" : times_cuda_fusedjtj,
            "mtFusedJTJTimes" : times_mt_fusedjtj,
            }

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/exp0029" + ".timing"
    pk.dump(costs, open(filename, "wb"))

# doTimingsExp0029('arap_mesh_deformation')
# doTimingsExp0029('image_warping')


# time opt for different example-sizes, for graph-stuff, this
# corresponds to numSubdivides, for pixelstuff it corresponds to the stride.
# materialization = [matfree|JTJ|fusedJTJ]
# TODO need separate lists for each backend and each materialization-strategy because cuda
# often runs out of memory
exp0002Sizes = {}
exp0002Sizes['arap_mesh_deformation'] = [1, 2, 3, 4, 5, 6]
# exp0002Sizes['arap_mesh_deformation'] = [1, 2, 3, 4, 5, 6, 7]
exp0002Sizes['cotangent_mesh_smoothing'] = [1, 2, 3, 4]
# exp0002Sizes['cotangent_mesh_smoothing'] = [1, 2, 3, 4, 5, 6]
# exp0002Sizes['embedded_mesh_deformation'] = [1, 2, 3, 4, 5] # >5 leads to out-of-memory error in cuda-backend(matfree)
# exp0002Sizes['embedded_mesh_deformation'] = [1, 2, 3, 4] # >4 leads to out-of-memory error in cuda-backend(jtj)
exp0002Sizes['embedded_mesh_deformation'] = [1, 2, 3] # >3 leads to out-of-memory error in cuda-backend(fusedjtj)
exp0002Sizes['image_warping'] = [6, 7, 8, 9, 10, 11, 12] # < 6 OOME for cuda-fusedjtj
exp0002Sizes['intrinsic_image_decomposition'] = [4,5,6,7,8,9,10] #<4 leads to OOME in cuda-fusedjtj
exp0002Sizes['optical_flow'] = [5, 6, 7,8,9] # <4 OOME in cuda-fusedjtj
exp0002Sizes['poisson_image_editing'] = [4, 5, 6, 7,8,9,10] # <4 OOME in cuda-fusedjtj
exp0002Sizes['robust_nonrigid_alignment'] = [1, 2, 3, 4]
exp0002Sizes['shape_from_shading'] = [1, 2, 3, 4]
# exp0002Sizes['volumetric_mesh_deformation'] = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
# exp0002Sizes['volumetric_mesh_deformation'] = [1, 2, 3, 4, 5, 6] # >6 throws out-of-mem error in cuda-backend(JTJ)
exp0002Sizes['volumetric_mesh_deformation'] = [1, 2, 3, 4, 5] # >5 throws out-of-mem error in cuda-backend(fusedJTJ)

expNumbers0002 = {'matfree' : {}, 'JTJ' : {}, 'fusedJTJ' : {}}
expNumbers0002['matfree']['backend_cpu'] = 2
expNumbers0002['JTJ']['backend_cpu'] = 3
expNumbers0002['fusedJTJ']['backend_cpu'] = 4

expNumbers0002['matfree']['backend_cpu_mt'] = 6
expNumbers0002['JTJ']['backend_cpu_mt'] = 7
expNumbers0002['fusedJTJ']['backend_cpu_mt'] = 8

expNumbers0002['matfree']['backend_cuda'] = 25
expNumbers0002['JTJ']['backend_cuda'] = 26
expNumbers0002['fusedJTJ']['backend_cuda'] = 27

exp0002FileArgs = {}
exp0002FileArgs['arap_mesh_deformation'] =         []
exp0002FileArgs['cotangent_mesh_smoothing'] =      []
exp0002FileArgs['embedded_mesh_deformation'] =     []
exp0002FileArgs['image_warping'] =                 ['--file', "2"]
exp0002FileArgs['intrinsic_image_decomposition'] = ['--file', '2']
exp0002FileArgs['optical_flow'] =                  ['--file', '2']
exp0002FileArgs['poisson_image_editing'] =         ['--file', '2']
exp0002FileArgs['volumetric_mesh_deformation'] =   []
def doTimingsExp000234(homedir, materialization, backend, numthreads):
    sizes = exp0002Sizes[homedir]

    if materialization not in ['matfree', 'JTJ', 'fusedJTJ']:
        errMsg = """
        doTimingsExp000234(): invalid materialization:

        {0}

        valid string-options are 'matfree', 'JTJ', 'fusedJTJ'
        """
        sys.exit(errMsg)

    if backend not in ['backend_cpu', 'backend_cpu_mt', 'backend_cuda']:
        errMsg = """
        doTimingsExp000234(): invalid backend:

        {0}

        valid string-options are 'backend_cpu', 'backend_cpu_mt', 'backend_cuda'
        """
        sys.exit(errMsg)

    # turn 'numthreads' into a string for later usage
    numthreads = str(numthreads)

    expNumber = expNumbers0002[materialization][backend]

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
        t_opt._args += ["--backend", backend, "--numthreads", numthreads]
        t_opt._args += exp0002FileArgs[homedir]

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
        t_opt._args.append("--noOutput")

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



    timingData = {
            "problemSizes" : problemSizes,
            "optPerSolveTimes" : perSolveTimes,
            "optPerNewtonTimes" : perNewtonTimes,
            "optPerLinIterTimes" : perLinIterTimes
            }
    print(timingData)

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/exp{:04d}".format(expNumber) + ".timing"



    pk.dump(timingData, open(filename, "wb"))



# define 3 sizes for each example, small-medium-large
exp0013Sizes = {}
exp0013Sizes['arap_mesh_deformation'] = [1, 2, 6]
exp0013Sizes['cotangent_mesh_smoothing'] = [0,1,5]
exp0013Sizes['embedded_mesh_deformation'] = [0,1,3]
exp0013Sizes['image_warping'] = [3,2,1]
exp0013Sizes['intrinsic_image_decomposition'] = [3,2,1]
exp0013Sizes['optical_flow'] = [3,2,1]
exp0013Sizes['poisson_image_editing'] = [3,2,1]
exp0013Sizes['robust_nonrigid_alignment'] = [0,1,2]
exp0013Sizes['shape_from_shading'] = [3,2,1]
exp0013Sizes['volumetric_mesh_deformation'] = [0,1,4]

expNumbers = [13, 14, 15]
# mt_numbers = [1,2,3,4] # need to change this, depending on the machine
mt_numbers = [1,2,3,4,5,6,7,8] # need to change this, depending on the machine
def doTimingsExp0013(homedir, sizeIndex):
    # probSize is 0, 1 or 2, depending on which size we want


    # if materialization not in ['matfree', 'JTJ', 'fusedJTJ']:
    #     errMsg = """
    #     doTimingsExp0013(): invalid materialization:

    #     {0}

    #     valid string-options are 'matfree', 'JTJ', 'fusedJTJ'
    #     """.format(materialization)
    #     sys.exit(errMsg)

    if sizeIndex not in [0,1,2]:
        errMsg = """
        doTimingsExp0013(): invalid sizeIndex:

        {0}

        valid integer-options are 0(=small), 1(=medium), 2(=large)
        """.format(sizeIndex)
        sys.exit(errMsg)


    sizes = exp0013Sizes[homedir]
    size = sizes[sizeIndex]
    expNumber = expNumbers[sizeIndex]


    tests = []

    # collect the size of the UnknownVector here
    # problemSizes = sizes
    problemSizes = []

    perSolveTimes = {}
    perNewtonTimes = {}
    perLinIterTimes = {}

    perSolNames = {}
    perNwtNames = {}
    perLinNames = {}

    for materialization in ['matfree', 'JTJ', 'fusedJTJ']:
        localPerSolveTimes = []
        localPerNewtonTimes = []
        localPerLinIterTimes = []
        for numthreads in mt_numbers:
            # create, define the example runs
            t_opt = ExampleRun(homedir)
            t_opt._printOutput = False
            t_opt._execCommand = "./" + homedir
            t_opt._args = ["--nIterations", "10", "--lIterations", "10"]
            t_opt._args += ["--backend", "backend_cpu_mt", "--numthreads", str(numthreads)]
            t_opt._args.append("--noOutput")

            if materialization == "matfree":
                pass # this is default
            elif materialization == "JTJ":
                t_opt._args.append("--useMaterializedJTJ")
            elif materialization =="fusedJTJ":
                t_opt._args.append("--useMaterializedJTJ")
                t_opt._args.append("--useFusedJTJ")
            else:
                pass # error checking is done at top of file


            # control size of the problem
            t_opt._args += [info.strideFlags[homedir], str(size)]

            # run them
            t_opt.run()

            # get final cost from raw text output
            print(t_opt._output)

            optPerSolveTime = prs.getAvgCategoryTimeOptMT(t_opt._output, 1, materialization, homedir)
            optPerNewtonTime = prs.getAvgCategoryTimeOptMT(t_opt._output, 2, materialization, homedir)
            optPerLinIterTime = prs.getAvgCategoryTimeOptMT(t_opt._output, 3, materialization, homedir)

            localPerSolveTimes.append(optPerSolveTime)
            localPerNewtonTimes.append(optPerNewtonTime)
            localPerLinIterTimes.append(optPerLinIterTime)

            unknownSize = prs.getUnknownSizeFromOutput(t_opt._output)
            problemSizes.append(unknownSize)

        perSolNames[materialization] = info.optPerSolveNames[materialization][homedir]
        perNwtNames[materialization] = info.optPerNewtonNames[materialization][homedir]
        perLinNames[materialization] = info.optPerLinIterNames[materialization][homedir]

        perSolveTimes[materialization] = localPerSolveTimes
        perNewtonTimes[materialization] = localPerNewtonTimes
        perLinIterTimes[materialization] = localPerLinIterTimes


    timingData = {
            "problemSizes" : problemSizes,
            "optPerSolveTimes" : perSolveTimes,
            "optPerSolveNames" : perSolNames,
            "optPerNewtonTimes" : perNewtonTimes,
            "optPerNewtonNames" : perNwtNames,
            "optPerLinIterTimes" : perLinIterTimes,
            "optPerLinIterNames" : perLinNames,
            "numThreads" : mt_numbers
            }
    print(timingData)

    # write the costs to a file so they can be read by the plotting
    # module
    filename = "./" + homedir + "/timings/exp00{0}".format(expNumber) + ".timing"



    pk.dump(timingData, open(filename, "wb"))

# doTimingsExp0013('arap_mesh_deformation', 2)

