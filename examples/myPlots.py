import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import myHelpers as hlp

# make plots pretty
plt.style.use('ggplot')

# define some arbitrary numbers for the plot (opSolve = Once Per Solve)



# plots per-solve times, per-newton timings and per-linearIter timings
# for a single example (e.g. for image_warping)
def ceresVsOptCpuBar(homedir, axarr=None, index=0):
    # 'data' is a dictionary with a pre-defined format.

    if axarr is None:
        fig, ax = plt.subplots()
    else:
        ax = axarr[index]

    data = pk.load(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))
    
    width = 0.1
    def myinds(center, lr):
        # if lr == -1, plot left bars
        # if lr == +1, plot as right bars
        # TODO verify inputs
        d = 0.05
        w = width

        if lr == -1:
            offset = -w/4
        elif lr == +1:
            offset = +w/4

        return center + np.array([-(w+d),0.0,(w+d)]) + offset


    def addExamplePerfCeres(ax, nums, c):
        # TODO verify inputs, make sure that we can stack stuff
        ax.bar(myinds(c,-1), nums, width/2, label='ceres')

    def addExamplePerfOpt(ax, nums, c):
        # TODO verify inputs, make sure that we can stack stuff
        ax.bar(myinds(c,+1), nums, width/2, label='opt')


    opt_lin = data['optPerLinIter']
    opt_newt = data['optPerNewton']
    opt_sol = data['optPerSolve']
    ceres_lin = data['ceresPerLinIter']
    ceres_newt = data['ceresPerNewton']

    t_opt = [opt_sol, opt_newt, opt_lin]
    t_ceres = [1, ceres_newt, ceres_lin]

    addExamplePerfOpt(ax, t_opt, 1)
    addExamplePerfCeres(ax, t_ceres, 1)


    # TODO set axis-labels
    ax.set_title(homedir)
    ax.legend()
    ax.set_xticks([0.85, 1.0, 1.15])
    ax.set_xticklabels(['per solver', 'per newton', 'per liniter'])
    # ax.semilogy()


    # save plot to file
    if axarr is None:
        plt.savefig(open("{0}/timings/ceresVsOptCpu.pdf".format(homedir), "wb"),
                format='pdf')

    # return (fig, ax)


    


expNumbers00234 = {'matfree' : {}, 'JTJ' : {}, 'fusedJTJ' : {}}
expNumbers00234['matfree']['backend_cpu'] = 2
expNumbers00234['JTJ']['backend_cpu'] = 3
expNumbers00234['fusedJTJ']['backend_cpu'] = 4

expNumbers00234['matfree']['backend_cpu_mt'] = 6
expNumbers00234['JTJ']['backend_cpu_mt'] = 7
expNumbers00234['fusedJTJ']['backend_cpu_mt'] = 8

expNumbers00234['matfree']['backend_cuda'] = 25
expNumbers00234['JTJ']['backend_cuda'] = 26
expNumbers00234['fusedJTJ']['backend_cuda'] = 27
def makePlotExp000234(homedir, materialization, backend, axarr=None, index=0):
    # timingData = {
    #         "problemSizes" : problemSizes,
    #         "optPerSolveTimes" : perSolveTimes,
    #         "optPerNewtonTimes" : perNewtonTimes,
    #         "optPerLinIterTimes" : perLinIterTimes
    #         }

    if axarr is None:
        fig, ax = plt.subplots()
    else:
        ax = axarr[index]

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

    expNumber = expNumbers00234[materialization][backend]

    data = pk.load(open("{0}/timings/exp{1:04d}.timing".format(homedir,expNumber)))

    sizes = data['problemSizes']
    t_sol = data['optPerSolveTimes']
    t_nwt = data['optPerNewtonTimes']
    t_lin = data['optPerLinIterTimes']


    # fig, ax = plt.subplots()

    ax.plot(sizes, t_sol, 'o-', label="per solve")
    ax.plot(sizes, t_nwt, 'o-', label="per newton")
    ax.plot(sizes, t_lin, 'o-', label="per linear")

    # reference for linear scaling
    refdata = np.array(sizes) * float(t_lin[0])/sizes[0] * 5
    ax.plot(sizes, refdata, '--', label="linear reference")

    if axarr is None:
        ax.legend()

    ax.set_title(homedir, fontsize=10, loc='left', verticalalignment='top')

    if axarr is None or index == 6 or index==7:
        ax.set_xlabel('unknown-size (bytes)')

    if axarr is None or index %2 ==0:
        ax.set_ylabel('time (ms)')

    ax.semilogx()
    ax.semilogy()

    if axarr is None:
        plt.savefig(open("{0}/timings/exp{1:04d}.pdf".format(homedir,expNumber), "wb"),
                format='pdf')

    # return fig, ax

    # plt.show()


# array with factors for bounds
exp0029BoundFactors = {}
# CHECK that these are consistent with table in thesis
exp0029BoundFactors['arap_mesh_deformation'] =         {'csr' : 1.96, 'ideal' : 4.08, 'matfree' : 0.65 }
exp0029BoundFactors['cotangent_mesh_smoothing'] =      {'csr' : 1.7, 'ideal' : 3.25, 'matfree' : 0.5 }
exp0029BoundFactors['embedded_mesh_deformation'] =     {'csr' : 3.2, 'ideal' : 6.22, 'matfree' : 1.13 }
exp0029BoundFactors['image_warping'] =                 {'csr' : 1.13, 'ideal' : 2.18, 'matfree' : 9.5 }
exp0029BoundFactors['intrinsic_image_decomposition'] = {'csr' : 1.18, 'ideal' : 2.23, 'matfree' : 3.4 }
exp0029BoundFactors['optical_flow'] =                  {'csr' : 1.12, 'ideal' : 2.21, 'matfree' : 2.4 }
exp0029BoundFactors['poisson_image_editing'] =         {'csr' : 1.14, 'ideal' : 2.11, 'matfree' : 12.8 }
exp0029BoundFactors['volumetric_mesh_deformation'] =   {'csr' : 1.1, 'ideal' : 2.1, 'matfree' : 5.2 }
def makePlotExp0029(homedir, axarr=None, index=0):
    # costs = {
    #         "unknownSizes" : unSizes,
    #         "plandataSizes" : unSizes,
    #         "cudaMatfreeTimes" : times_cuda_matfree,
    #         "cudaFusedJTJTimes" : times_cuda_matfree,
    #         "mtFusedJTJTimes" : times_cuda_matfree,
    #         }

    if axarr is None:
        fig, ax = plt.subplots()
    else:
        ax = axarr[index]


    expNumber = 29

    data = pk.load(open("{0}/timings/exp{1:04d}.timing".format(homedir,expNumber)))

    unSizes = np.array(data['unknownSizes'])
    pdSizes = np.array(data['plandataSizes'])
    times_cuda_matfree = np.array(data['cudaMatfreeTimes'])
    times_cuda_fusedjtj = np.array(data['cudaFusedJTJTimes'])
    times_mt_fusedjtj = np.array(data['mtFusedJTJTimes'])


    # fig, ax = plt.subplots()

    # plot data
    ax.plot(unSizes, times_cuda_matfree, 'o-', label='cuda matfree')
    ax.plot(unSizes, times_cuda_fusedjtj, 'o-', label='cuda fusedjtj')
    ax.plot(unSizes, times_mt_fusedjtj, 'o-', label='mt4 fusedjtj')

    # compute a lower bound for mt4 performance: plan-data has
    bd_csr     = times_mt_fusedjtj / exp0029BoundFactors[homedir]['csr']
    bd_ideal   = times_mt_fusedjtj / exp0029BoundFactors[homedir]['ideal']
    bd_matfree = times_mt_fusedjtj / exp0029BoundFactors[homedir]['matfree']
    ax.plot(unSizes, bd_csr, '--', label='low csr')
    ax.plot(unSizes, bd_ideal, '--', label='low ideal')
    ax.plot(unSizes, bd_matfree, '--', label='low matfree')



    if axarr is None:
        ax.legend()

    ax.set_title(homedir, fontsize=10, loc='left', verticalalignment='top')

    if axarr is None or index in [6,7]:
        ax.set_xlabel('unknown-size (bytes)')

    if axarr is None or index in [0,2,4,6]:
        ax.set_ylabel('avg. time (ms)')

    ax.semilogx()
    ax.semilogy()

    if axarr is None:
        plt.savefig(open("{0}/timings/exp{1:04d}.pdf".format(homedir,expNumber), "wb"),
                format='pdf')

# makePlotExp0029('arap_mesh_deformation')


def makePlotExp0028(homedir, axarr=None, index=0):
    # costs = {
    #         "unknownSizes" : unSizes,
    #         "planDataSizes" : pdSizes,
    #         }

    # if materialization not in ['matfree', 'JTJ', 'fusedJTJ']:
    #     errMsg = """
    #     doTimingsExp000234(): invalid materialization:

    #     {0}

    #     valid string-options are 'matfree', 'JTJ', 'fusedJTJ'
    #     """
    #     sys.exit(errMsg)

    if axarr is None:
        fig, ax = plt.subplots()
    else:
        ax = axarr[index]


    expNumber = 28

    data = pk.load(open("{0}/timings/exp{1:04d}.timing".format(homedir,expNumber)))

    pdSizes = data['planDataSizes']
    unSizes = data['unknownSizes']


    # fig, ax = plt.subplots()

    # plot data
    facs = {'matfree' : [], 'JTJ' : [], 'fusedJTJ' : []}
    for mater in ['matfree', 'JTJ', 'fusedJTJ']:
        ax.plot(unSizes[mater], pdSizes[mater], 'o-', label=mater)

        # calculate scaling factor (not visible in logplot) and print it on the plot
        # (linear regression with single unknown) --> lower-right corner
        fac = float(np.dot(unSizes[mater],pdSizes[mater]))/np.dot(unSizes[mater], unSizes[mater])
        facs[mater] = fac

    ax.annotate("slope(matfree) = {:4.1f}".format(facs['matfree']), xy = (0.95,0.15), xycoords='axes fraction', horizontalalignment='right', fontsize=7)
    ax.annotate("slope(jtj) = {:4.1f}".format(facs['JTJ']), xy = (0.95,0.1), xycoords='axes fraction', horizontalalignment='right', fontsize=7)
    ax.annotate("slope(fusedjtj) = {:4.1f}".format(facs['fusedJTJ']), xy = (0.95,0.05), xycoords='axes fraction', horizontalalignment='right', fontsize=7)

    # plot linear regression for reference (slightly  upscaled so that plots don't lie on top fo each other
    refdata = 10 * facs['matfree'] * np.array(unSizes['matfree'])
    ax.plot(unSizes['matfree'], refdata, '--', label="lin. scal. reference")

    # plot gpu memory limit (geforce gtx 680 --> 2 GB)
    gpuMem = 2048*1e6
    gpuMems = gpuMem * np.ones(len(pdSizes['matfree'])) 
    ax.plot(unSizes['matfree'], gpuMems, '--', color='black', alpha=0.5, linewidth=0.5, label='gpu mem')

    # extrapolate unknownsize at which we run out of gpu memory
    # do for all materializations and annotate plot
    maxUnSizes = {'matfree' : [], 'JTJ' : [], 'fusedJTJ' : []}
    for mater in ['matfree', 'JTJ', 'fusedJTJ']:
        maxUnSizes[mater] = int(gpuMem/facs[mater])/(1024*1024)

    ax.annotate("maxUnknownSize(matfree) = {:4.1f} MB".format(maxUnSizes['matfree']), xy = (0.05,0.8), xycoords='axes fraction', horizontalalignment='left', fontsize=7)
    ax.annotate("maxUnknownSize(jtj) = {:4.1f} MB".format(maxUnSizes['JTJ']), xy = (0.05,0.75), xycoords='axes fraction', horizontalalignment='left', fontsize=7)
    ax.annotate("maxUnknownSize(fusedjtj) = {:4.1f} MB".format(maxUnSizes['fusedJTJ']), xy = (0.05,0.7), xycoords='axes fraction', horizontalalignment='left', fontsize=7)




    # other formatting stuff
    if axarr is None:
        lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        pass

    ax.set_title(homedir, fontsize=10, verticalalignment='top', loc='left')

    if axarr is not None:
        ax.tick_params(labelsize=7)

    if axarr is not  None and (index==6 or index==7):
        ax.set_xlabel('unknown-size (bytes)', fontsize=10)

    if axarr is not None and index % 2 == 0:
        ax.set_ylabel('plandata-size (bytes)', fontsize=10)

    ax.semilogx()
    ax.semilogy()

    # plt.savefig(open("{0}/timings/exp{1:04d}.pdf".format(homedir,expNumber), "wb"),
    #         format='pdf')
    if axarr is None:
        plt.savefig(open("{0}/timings/exp{1:04d}.pdf".format(homedir,expNumber), "wb"),
                bbox_extra_artists=[lgd],
                bbox_inches='tight',
                format='pdf')

    # return fig, ax

    # plt.show()

# makePlotExp00028('arap_mesh_deformation')



expNumbers = {'matfree' : {}, 'JTJ' : {}, 'fusedJTJ' : {}}
expNumbers['matfree'] = 22
expNumbers['JTJ'] = 23
expNumbers['fusedJTJ'] = 24

expNumbersCpu = {'matfree' : {}, 'JTJ' : {}, 'fusedJTJ' : {}}
expNumbersCpu['matfree'] = 2
expNumbersCpu['JTJ'] = 3
expNumbersCpu['fusedJTJ'] = 4

expNumbersMt = {'matfree' : {}, 'JTJ' : {}, 'fusedJTJ' : {}}
expNumbersMt['matfree'] = 6
expNumbersMt['JTJ'] = 7
expNumbersMt['fusedJTJ'] = 8
def makePlotExp0022(homedir, materialization, axarr=None, index=0):
    # timingData = {
    #         "problemSizes" : problemSizes,
    #         "optPerSolveTimes" : perSolveTimes,
    #         "optPerNewtonTimes" : perNewtonTimes,
    #         "optPerLinIterTimes" : perLinIterTimes
    #         }

    if axarr is None:
        fig, ax = plt.subplots()
    else:
        ax = axarr[index]

    if materialization not in ['matfree', 'JTJ', 'fusedJTJ']:
        errMsg = """
        doTimingsExp0022(): invalid materialization:

        {0}

        valid string-options are 'matfree', 'JTJ', 'fusedJTJ'
        """
        sys.exit(errMsg)


    expNumber = expNumbers[materialization]
    expNumberCpu = expNumbersCpu[materialization]
    expNumberMt = expNumbersMt[materialization]

    data_cpu = pk.load(open("{0}/timings/exp000{1}.timing".format(homedir,expNumberCpu)))
    data_mt = pk.load(open("{0}/timings/exp000{1}.timing".format(homedir,expNumberMt)))

    sizes_cpu = data_cpu['problemSizes']
    t_sol_cpu = data_cpu['optPerSolveTimes']
    t_nwt_cpu = data_cpu['optPerNewtonTimes']
    t_lin_cpu = data_cpu['optPerLinIterTimes']

    sizes_mt = data_mt['problemSizes']
    t_sol_mt = data_mt['optPerSolveTimes']
    t_nwt_mt = data_mt['optPerNewtonTimes']
    t_lin_mt = data_mt['optPerLinIterTimes']


    # fig, ax = plt.subplots()

    ax.plot(sizes_cpu, t_sol_cpu, 'o-', label="per solve cpu", color='C0')
    ax.plot(sizes_mt, t_sol_mt, 'x-', label="per solve mt", color='C0')

    ax.plot(sizes_cpu, t_nwt_cpu, 'o-', label="per newton cpu", color='C1')
    ax.plot(sizes_mt, t_nwt_mt, 'x-', label="per newton mt", color='C1')

    ax.plot(sizes_cpu, t_lin_cpu, 'o-', label="per linear cpu", color='C2')
    ax.plot(sizes_mt, t_lin_mt, 'x-', label="per linear mt", color='C2')

    if axarr is None:
        ax.legend()

    ax.set_title(homedir, fontsize=10, loc='left', verticalalignment='top')

    if axarr is None or index in [6,7]:
        ax.set_xlabel('unknown-size (bytes)')

    if axarr is None or index in [0,2,4,6]:
        ax.set_ylabel('time (ms)')

    ax.semilogx()
    ax.semilogy()

    if axarr is None:
        plt.savefig(open("{0}/timings/exp{1:04d}.pdf".format(homedir,expNumber), "wb"),
                format='pdf')

    # return fig, ax

    # plt.show()
    

# create bar-plot where for each example we have matfree vs. jtj vs. fusedjtj
def makePlotExp0005(examples):
    # timingData = {
    #         "problemSizes" : problemSizes,
    #         "optPerSolveTimes" : perSolveTimes,
    #         "optPerNewtonTimes" : perNewtonTimes,
    #         "optPerLinIterTimes" : perLinIterTimes
    #         }

    # see usage below for meaning of the arguments.
    def addNextBarPlot(t1, t2, t3, t4, t5, t6, t7, t8, t9, center, theLabel):
        # center should be 1, 2, 3, 4, 5, etc.
        width = 0.05
        vals = [t1, t2, t3, t4, t5, t6, t7, t8, t9]

        # normalize data
        vals = np.array(vals)
        vals = 100*vals/np.max(vals)

        # x-location of the bars relative to the center
        rel_xs = np.array([-5.0, -4.0, -3.0,
            -1.0, 0.0, 1.0,
            3.0, 4.0, 5.0])
        xs = center + rel_xs*width
        ax.bar(xs, vals, width, label=theLabel)

        # annotate example with indicator. The indicator compares
        # matfree to jtj. It calculates how many seconds are added
        # from matfree to jtj in per-newton-time and how many
        # seconds are saved in per-linear-time. The ratio then tells us
        # the minimum amount of linear iterations that are necessary to
        # save time overall. per-solve time is not considered in the calculation.
        # WE do the same for matfree vs. fusedjtj

        # first matfree vs jtj (per-newton)
        per_nwt_extra = t5-t2
        per_lin_savings = t3-t6

        min_iter = per_nwt_extra/per_lin_savings
        min_iter_str = "n*{:02.1f}".format(min_iter)

        ax.annotate(min_iter_str, (center,80))

        # then matfree vs fusedjtj (per-newton, plotted below the previous)
        per_nwt_extra = t8-t2
        per_lin_savings = t3-t9

        min_iter = per_nwt_extra/per_lin_savings
        min_iter_str = "n*{:02.1f}".format(min_iter)

        ax.annotate(min_iter_str, (center,70))

        # first matfree vs jtj (per-solve)
        per_nwt_extra = t4-t1
        per_lin_savings = t3-t6

        min_iter = per_nwt_extra/per_lin_savings
        min_iter_str = "{:02.1f}".format(min_iter)

        ax.annotate(min_iter_str, (center,50))

        # then matfree vs fusedjtj (per-newton, plotted below the previous)
        per_nwt_extra = t7-t1
        per_lin_savings = t3-t9

        min_iter = per_nwt_extra/per_lin_savings
        min_iter_str = "{:02.1f}".format(min_iter)

        ax.annotate(min_iter_str, (center,40))


    # TODO for now, we just take the largest time-value and assume that
    # it belongs to the largest example, this should be fixed at some point.
    def getTimeForLargestExample(times):
        # sizes = np.array(sizes)
        # times = np.array(times)

        # max_location = np.argmax(sizes)
        # return times[max_location]
        times = np.array([times])
        return np.max(times)



    fig, ax = plt.subplots()
    fig.set_size_inches(*hlp.paperSizes['A5_landscape'])

    exmpl_center = 1
    for homedir in examples:

        data_matfree = pk.load(open("{0}/timings/exp0002.timing".format(homedir)))
        data_jtj = pk.load(open("{0}/timings/exp0003.timing".format(homedir)))
        data_fusedjtj = pk.load(open("{0}/timings/exp0004.timing".format(homedir)))

        t_sol_matfree = getTimeForLargestExample(data_matfree['optPerSolveTimes'])
        t_sol_jtj = getTimeForLargestExample(data_jtj['optPerSolveTimes'])
        t_sol_fusedjtj = getTimeForLargestExample(data_fusedjtj['optPerSolveTimes'])

        t_nwt_matfree = getTimeForLargestExample(data_matfree['optPerNewtonTimes'])
        t_nwt_jtj = getTimeForLargestExample(data_jtj['optPerNewtonTimes'])
        t_nwt_fusedjtj = getTimeForLargestExample(data_fusedjtj['optPerNewtonTimes'])

        t_lin_matfree = getTimeForLargestExample(data_matfree['optPerLinIterTimes'])
        t_lin_jtj = getTimeForLargestExample(data_jtj['optPerLinIterTimes'])
        t_lin_fusedjtj = getTimeForLargestExample(data_fusedjtj['optPerLinIterTimes'])

        addNextBarPlot(t_sol_matfree, t_nwt_matfree, t_lin_matfree,
                t_sol_jtj, t_nwt_jtj, t_lin_jtj,
                t_sol_fusedjtj, t_nwt_fusedjtj, t_lin_fusedjtj,
                exmpl_center, homedir)
        exmpl_center += 1



    lgd = ax.legend(loc='upper center', ncol=3,
            bbox_to_anchor=(0.5,-0.02), fontsize=7)
    ax.tick_params(bottom='off', labelbottom='off')

    ax.set_ylabel('time (ms)')
    # ax.semilogy()

    plt.savefig(open("./timings/exp0005.pdf", "wb"),
            bbox_extra_artists=[lgd],
            bbox_inches='tight',
            format='pdf')

    # return fig, ax

    # plt.show()



# create scaling-plot w.r.t. numthreads
def makePlotExp0013(homedir, expNumber, fig=None, axarr=None, index=0):
    # timingData = {
    #         "problemSizes" : problemSizes,
    #         "optPerSolveTimes" : perSolveTimes,
    #         "optPerSolveNames" : perSolNames,
    #         "optPerNewtonTimes" : perNewtonTimes,
    #         "optPerNewtonNames" : perNwtNames,
    #         "optPerLinIterTimes" : perLinIterTimes,
    #         "optPerLinIterNames" : perLinNames,
    #         "numThreads" : mt_numbers
    #         }


    if axarr is None:
        fig, ax = plt.subplots()
    else:
        ax = axarr[index]
        

    data = pk.load(open("{0}/timings/exp00{1}.timing".format(homedir,expNumber)))

    # set a different marker for each materialization
    markers = {}
    markers['matfree'] = 'o'
    markers['JTJ'] = 'x'
    markers['fusedJTJ'] = 's'

    lgd_names = {}
    lgd_names['matfree'] = 'MF'
    lgd_names['JTJ'] = 'JTJ'
    lgd_names['fusedJTJ'] = 'FJTJ'

    # create annotation for kernel-names
    name_offset = {}
    name_offset['matfree'] = -0.1
    name_offset['JTJ'] = -0.2
    name_offset['fusedJTJ'] = -0.3

    names_annotations = []

    names_sol = data['optPerSolveNames']
    names_nwt = data['optPerNewtonNames']
    names_lin = data['optPerLinIterNames']

    name_an = """
    matfree SLV: {0}
    matfree NWT: {1}
    matfree LIN: {2}

    JTJ     SLV: {3}
    JTJ     NWT: {4}
    JTJ     LIN: {5}

    FJTJ    SLV: {6}
    FJTJ    NWT: {7}
    FJTJ    LIN: {8}
    """.format(names_sol['matfree'], names_nwt['matfree'], names_lin['matfree'],
            names_sol['JTJ'], names_nwt['JTJ'], names_lin['JTJ'],
            names_sol['fusedJTJ'], names_nwt['fusedJTJ'], names_lin['fusedJTJ'])

    for materialization in ['matfree', 'JTJ', 'fusedJTJ']:

        t_sol = data['optPerSolveTimes'][materialization]
        t_nwt = data['optPerNewtonTimes'][materialization]
        t_lin = data['optPerLinIterTimes'][materialization]
        nthreads = np.array(data['numThreads'])

            

        ax.plot(nthreads, t_sol, label='per solve '+lgd_names[materialization], marker=markers[materialization], color='C0')
        ax.plot(nthreads, t_nwt, label='per newton '+lgd_names[materialization], marker=markers[materialization], color='C1')
        ax.plot(nthreads, t_lin, label='per linear '+lgd_names[materialization], marker=markers[materialization], color='C2')

        # plot an ideal-scaling curve with each plot for visual reference.
        ideal_sol = t_sol[0]/nthreads
        ideal_nwt = t_nwt[0]/nthreads
        ideal_lin = t_lin[0]/nthreads
        ax.plot(nthreads, ideal_lin, '--', linewidth=0.2, alpha=0.3, color='black')
        ax.plot(nthreads, ideal_sol, '--', linewidth=0.2, alpha=0.3, color='black')
        ax.plot(nthreads, ideal_nwt, '--', linewidth=0.2, alpha=0.3, color='black')

        # write each number into the plot for now, just for easier data-evaluation.
        if axarr is None:
            for t in [t_sol, t_nwt, t_lin]:
                for k in range(len(nthreads)):
                    ax.annotate(str(t[k]), xy=(nthreads[k], t[k]), size=5)



    problemSize = data['problemSizes'][0]

    ax.semilogy()
    ax.yaxis.grid(True, which='minor')

    # write problemsize (bytes) in lower-left  corner:
    if axarr is None:
        ax.annotate("prob-size: {0} bytes".format(problemSize), xycoords='axes fraction', xy=(0.05,0.05),
                fontsize=7)

    # write down the names of all kernels involved in e.g. per-newton timings
    if axarr is None:
        an = ax.annotate(name_an, xycoords='axes fraction', xy=(0,-0.1),
                fontsize=7, verticalalignment='top', fontname='monospace')


    # lgd = ax.legend()
    if axarr is None:
        lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),
                fontsize=7)

    if axarr is None or index in [6,7]:
        ax.set_xlabel('num. of threads')

    if axarr is None or index in [0,2,4,6]:
        ax.set_ylabel('avg. time (ms)')

    if axarr is None:
        ax.set_title(homedir, fontsize=7)
    else:
        ax.annotate(homedir, fontsize=10, xy=(0.05,0.05), xycoords='axes fraction')

    # plt.savefig(open("./{0}/timings/exp00{1}.pdf".format(homedir,expNumber), "wb"),
    #         bbox_extra_artists=[lgd,an],
    #         bbox_inches='tight',
    #         format='pdf')

    if axarr is None:
        plt.savefig(open("./{0}/timings/exp00{1}.pdf".format(homedir,expNumber), "wb"),
                bbox_extra_artists=[lgd,an],
                bbox_inches='tight',
                format='pdf')

    pk.dump(fig, open("./{0}/timings/exp00{1}.plot".format(homedir,expNumber), "wb"))

    # return fig, ax

    # plt.show()

# makePlotExp0013('arap_mesh_deformation')

# this is how to load a pickle file and use it to create a plot
def test():
    theData = pk.load(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))
    fig, ax = ceresVsOptCpuBar(theData)

    # save plot to file
    plt.savefig(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))


