# test 3-category scalability of opt-matfree w.r.t. problem-size
from myPlots import makePlotExp0022
from myArgParsers import experimentParser
from PyPDF2 import PdfFileMerger
import myHelpers as hlp
import matplotlib.pyplot as plt
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
    print("exp0024.py: Not performing timings for this experiment. Using results from experiments 2,3,4,6,7,8")

if exParser._doPlotAll:
    finalfig, axarr = plt.subplots(4,2)
    finalfig.set_size_inches(*hlp.paperSizes['A4'])
    axarr = axarr.reshape(-1)
    counter=0
    for homedir in folders:
        print("exp0024.py: creating plot for {0}".format(homedir))

        # first save the individual pdfplot
        makePlotExp0022(homedir, 'fusedJTJ')

        makePlotExp0022(homedir, 'fusedJTJ', axarr=axarr, index=counter)
        counter += 1

    finalfig.legend(ncol=6, loc='lower center',
            fontsize=7, bbox_to_anchor=(0.45,0.02))
    finalfig.suptitle('Opt-cpu vs. Opt-mt(1) fusedjtj', fontsize=15, y=0.92)
    finalfig.savefig(open("./timings/exp0024_singlepage.pdf", "wb"),
            bbox_inches='tight',
            format='pdf')

    merger = PdfFileMerger()
    for homedir in folders:
        # then append it to the Merger
        merger.append(open("{0}/timings/exp0024.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0024.pdf")


if exParser._exampleName is not None:
    homedir = exParser._exampleName

    print("exp0024.py: Not performing timings for this experiment. Using results from experiments 2,3,4,6,7,8")

    print("exp0024.py: creating plot for {0}".format(homedir))
    makePlotExp0022(homedir, 'fusedJTJ')

    # re-do the file with the collected plots
    merger = PdfFileMerger()
    for homedir in folders:
        merger.append(open("{0}/timings/exp0024.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0024.pdf")