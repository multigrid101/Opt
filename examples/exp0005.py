# Plot matfree vs. JTJ vx. fusedJTJ for all examples.
# Uses data from exp0002/3/4
# WARNING: Simulations will take a long time because they will re-do all
# simulations for experiments 0002/3/4
from myPlots import makePlotExp0005
from myTimings import doTimingsExp000234
from myArgParsers import experimentParser
import sys


folders = []
folders.append("arap_mesh_deformation")

folders.append("cotangent_mesh_smoothing")

folders.append("embedded_mesh_deformation")

folders.append("image_warping")

folders.append("intrinsic_image_decomposition")

folders.append("optical_flow")

folders.append("poisson_image_editing")

# This example is broken at the moment
# folders.append("robust_nonrigid_alignment")

# TODO need to somehow do extra timings for this example, we only
# need results for one problem-size
# --> cannot control size for this example
# folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")


exParser = experimentParser()
exParser.parse_args(len(sys.argv), sys.argv)



if exParser._doTimeAll:
    for homedir in folders:
        print("exp0005.py: performing simulations for {0}".format(homedir))
        doTimingsExp000234(homedir, 'matfree')
        doTimingsExp000234(homedir, 'matfree')
        doTimingsExp000234(homedir, 'matfree')

if exParser._doPlotAll:
    print("exp0005.py: creating plot")
    makePlotExp0005(folders)

if exParser._exampleName is not None:
    errMsg = """

    exp0005.py: This is a plot that collects data from ALL examples,
        so using an example-name is illegal here. Only 'plot' and 'time'
        are valid actions for this file.

    """
