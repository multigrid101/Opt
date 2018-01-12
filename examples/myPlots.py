import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

# make plots pretty
plt.style.use('ggplot')

# define some arbitrary numbers for the plot (opSolve = Once Per Solve)



# plots per-solve times, per-newton timings and per-linearIter timings
# for a single example (e.g. for image_warping)
def ceresVsOptCpuBar(homedir):
    # 'data' is a dictionary with a pre-defined format.

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
    ceres_lin = data['ceresPerLinIter']
    ceres_newt = data['ceresPerNewton']

    t_opt = [1, opt_newt, opt_lin]
    t_ceres = [1, ceres_newt, ceres_lin]

    fig, ax = plt.subplots()

    addExamplePerfOpt(ax, t_opt, 1)
    addExamplePerfCeres(ax, t_ceres, 1)

    # TODO set axis-labels
    ax.set_title(homedir)
    ax.legend()


    # save plot to file
    plt.savefig(open("{0}/timings/ceresVsOptCpu.pdf".format(homedir), "wb"),
            format='pdf')

    # return (fig, ax)


expNumbers = {'matfree' : {}, 'JTJ' : {}, 'fusedJTJ' : {}}
expNumbers['matfree']['backend_cpu'] = 2
expNumbers['JTJ']['backend_cpu'] = 3
expNumbers['fusedJTJ']['backend_cpu'] = 4

expNumbers['matfree']['backend_cpu_mt'] = 6
expNumbers['JTJ']['backend_cpu_mt'] = 7
expNumbers['fusedJTJ']['backend_cpu_mt'] = 8
def makePlotExp000234(homedir, materialization, backend):
    # timingData = {
    #         "problemSizes" : problemSizes,
    #         "optPerSolveTimes" : perSolveTimes,
    #         "optPerNewtonTimes" : perNewtonTimes,
    #         "optPerLinIterTimes" : perLinIterTimes
    #         }

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

    expNumber = expNumbers[materialization][backend]

    data = pk.load(open("{0}/timings/exp000{1}.timing".format(homedir,expNumber)))

    sizes = data['problemSizes']
    t_sol = data['optPerSolveTimes']
    t_nwt = data['optPerNewtonTimes']
    t_lin = data['optPerLinIterTimes']


    fig, ax = plt.subplots()

    ax.plot(sizes, t_sol, 'o-', label="per solve")
    ax.plot(sizes, t_nwt, 'o-', label="per newton")
    ax.plot(sizes, t_lin, 'o-', label="per linear")

    ax.legend()
    ax.set_title(homedir)
    ax.set_xlabel('unknown-size (bytes)')
    ax.set_ylabel('time (ms)')

    plt.savefig(open("{0}/timings/exp000{1}.pdf".format(homedir,expNumber), "wb"),
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



    lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.savefig(open("./timings/exp0005.pdf", "wb"),
            bbox_extra_artists=[lgd],
            bbox_inches='tight',
            format='pdf')

    # return fig, ax

    # plt.show()



# this is how to load a pickle file and use it to create a plot
def test():
    theData = pk.load(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))
    fig, ax = ceresVsOptCpuBar(theData)

    # save plot to file
    plt.savefig(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))


