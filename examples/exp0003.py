# test 3-category scalability of opt-matfree w.r.t. problem-size
from myPlots import makePlotExp000234
from myTimings import doTimingsExp000234
from myArgParsers import experimentParser
from PyPDF2 import PdfFileMerger
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

# cannot control size for this example
# folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")


exParser = experimentParser()
exParser.parse_args(len(sys.argv), sys.argv)



if exParser._doTimeAll:
    for homedir in folders:
        print("exp0003.py: performing simulations for {0}".format(homedir))
        doTimingsExp000234(homedir, 'JTJ', 'backend_cpu', 1)

if exParser._doPlotAll:
    for homedir in folders:
        print("exp0003.py: creating plot for {0}".format(homedir))
        makePlotExp000234(homedir, 'JTJ', 'backend_cpu')

    merger = PdfFileMerger()
    for homedir in folders:
        # then append it to the Merger
        merger.append(open("{0}/timings/exp0003.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0003.pdf")

if exParser._exampleName is not None:
    homedir = exParser._exampleName

    print("exp0003.py: performing simulations for {0}".format(homedir))
    doTimingsExp000234(exParser._exampleName, 'JTJ', 'backend_cpu', 1)

    print("exp0003.py: creating plot for {0}".format(homedir))
    makePlotExp000234(exParser._exampleName, 'JTJ', 'backend_cpu')

    # re-do the file with the collected plots
    merger = PdfFileMerger()
    for homedir in folders:
        merger.append(open("{0}/timings/exp0003.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0003.pdf")
