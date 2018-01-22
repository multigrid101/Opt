# test 3-category scalability of opt-matfree w.r.t. problem-size
from myPlots import makePlotExp0028
from myTimings import doTimingsExp0028
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
        print("exp0028.py: performing simulations for {0}".format(homedir))
        doTimingsExp0028(homedir)

if exParser._doPlotAll:
    # finalfig, axarr = plt.subplots(4,2, sharex=True)
    finalfig, axarr = plt.subplots(4,2)
    finalfig.set_size_inches(*hlp.paperSizes['A4'])
    axarr = axarr.reshape(-1)
    counter=0
    for homedir in folders:
        print("exp0028.py: creating plot for {0}".format(homedir))

        # first save the individual pdfplot
        if os.path.isfile("{0}/timings/exp0028.timing".format(homedir)):
            makePlotExp0028(homedir)

            makePlotExp0028(homedir, axarr=axarr, index=counter)
            counter += 1

    finalfig.legend(ncol=5, loc='lower center',
            fontsize=7, bbox_to_anchor=(0.5,0.02))
    finalfig.suptitle('Memory Requirements', fontsize=15, y=0.92)
    finalfig.savefig(open("./timings/exp0028_singlepage.pdf", "wb"),
            bbox_inches='tight',
            format='pdf')

    hlp.mergePdfsOfAvailableFiles("./timings/exp0028.pdf", folders, "{0}/timings/exp0028.pdf")


if exParser._exampleName is not None:
    homedir = exParser._exampleName

    print("exp0028.py: performing simulations for {0}".format(homedir))
    doTimingsExp0028(homedir)

    print("exp0028.py: creating plot for {0}".format(homedir))
    makePlotExp0028(homedir)

    hlp.mergePdfsOfAvailableFiles("./timings/exp0028.pdf", folders, "{0}/timings/exp0028.pdf")
