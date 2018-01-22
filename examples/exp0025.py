# test 3-category scalability of opt-matfree w.r.t. problem-size
from myPlots import makePlotExp000234
from myTimings import doTimingsExp000234
from myArgParsers import experimentParser
import myHelpers as hlp
import matplotlib.pyplot as plt

from PyPDF2 import PdfFileMerger
import os.path
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
        print("exp0025.py: performing simulations for {0}".format(homedir))
        doTimingsExp000234(homedir, 'matfree', 'backend_cuda', 1)

if exParser._doPlotAll:
    finalfig, axarr = plt.subplots(4,2)
    finalfig.set_size_inches(*hlp.paperSizes['A4'])
    axarr = axarr.reshape(-1)
    counter=0
    for homedir in folders:
        print("exp0025.py: creating plot for {0}".format(homedir))

        # first save the individual pdfplot
        makePlotExp000234(homedir, 'matfree', 'backend_cuda')

        makePlotExp000234(homedir, 'matfree', 'backend_cuda', axarr=axarr, index=counter)
        counter += 1

    finalfig.legend(ncol=5, loc='lower center',
            fontsize=7, bbox_to_anchor=(0.5,0.02))
    finalfig.suptitle('Opt-gpu matrix-free', fontsize=15, y=0.92)
    finalfig.savefig(open("./timings/exp0025_singlepage.pdf", "wb"),
            bbox_inches='tight',
            format='pdf')

    hlp.mergePdfsOfAvailableFiles("./timings/exp0025.pdf", folders, "{0}/timings/exp0025.pdf")


if exParser._exampleName is not None:
    homedir = exParser._exampleName

    print("exp0025.py: performing simulations for {0}".format(homedir))
    doTimingsExp000234(exParser._exampleName, 'matfree', 'backend_cuda', 1)

    print("exp0025.py: creating plot for {0}".format(homedir))
    makePlotExp000234(exParser._exampleName, 'matfree', 'backend_cuda')

    hlp.mergePdfsOfAvailableFiles("./timings/exp0025.pdf", folders, "{0}/timings/exp0025.pdf")
