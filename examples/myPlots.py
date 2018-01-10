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
        ax.bar(myinds(c,-1), nums, width/2)

    def addExamplePerfOpt(ax, nums, c):
        # TODO verify inputs, make sure that we can stack stuff
        ax.bar(myinds(c,+1), nums, width/2)


    opt_lin = data['optPerLinIter']
    opt_newt = data['optPerNewton']
    ceres_lin = data['ceresPerLinIter']
    ceres_newt = data['ceresPerNewton']

    t_opt = [1, opt_newt, opt_lin]
    t_ceres = [1, ceres_newt, ceres_lin]

    fig, ax = plt.subplots()

    addExamplePerfOpt(ax, t_opt, 1)
    addExamplePerfCeres(ax, t_ceres, 1)


    # save plot to file
    plt.savefig(open("{0}/timings/ceresVsOptCpu.pdf".format(homedir), "wb"))

    # return (fig, ax)
    



# this is how to load a pickle file and use it to create a plot
def test():
    theData = pk.load(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))
    fig, ax = ceresVsOptCpuBar(theData)

    # save plot to file
    plt.savefig(open("{0}/timings/ceresVsOptCpu.timing".format(homedir)))


