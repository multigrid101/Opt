# test 3-category scalability of opt-matfree w.r.t. problem-size
from myPlots import makePlotExp0013
from myTimings import doTimingsExp0013
from myArgParsers import experimentParser
from PyPDF2 import PdfFileMerger
import sys
import matplotlib.pyplot as plt
import myHelpers as hlp
# import pickle as pk


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
        print("exp0013.py: performing simulations for {0}".format(homedir))
        doTimingsExp0013(homedir, 0)

if exParser._doPlotAll:
    finalfig, axarr = plt.subplots(4,2, sharex=True)
    finalfig.set_size_inches(*hlp.paperSizes['A4'])
    axarr = axarr.reshape(-1)
    counter=0
    for homedir in folders:
        print("exp0013.py: creating plot for {0}".format(homedir))

        # first save the individual pdfplot
        makePlotExp0013(homedir, 13)

        # add plot to single-page plot
        makePlotExp0013(homedir, 13, fig=finalfig, axarr=axarr, index=counter)
        counter += 1

    # plt.show()
    finalfig.legend(ncol=3, loc='lower center',
            fontsize=7, bbox_to_anchor=(0.45,0.01))
    finalfig.suptitle('Opt-mt(n) matrix-free strong scaling small size', fontsize=15, y=0.92)
    finalfig.savefig(open("./timings/exp0013_singlepage.pdf", "wb"),
            bbox_inches='tight',
            format='pdf')

    merger = PdfFileMerger()
    for homedir in folders:
        # then append it to the Merger
        merger.append(open("{0}/timings/exp0013.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0013.pdf")

    # load all plots and merge them in a single-page pdf.
    # axarr[1].plot([1,2,3,4])

    # finalfig.show()
    # plt.show()


if exParser._exampleName is not None:
    homedir = exParser._exampleName

    print("exp0013.py: performing simulations for {0}".format(homedir))
    doTimingsExp0013(exParser._exampleName, 0)

    print("exp0013.py: creating plot for {0}".format(homedir))
    makePlotExp0013(exParser._exampleName, 13)

    # re-do the file with the collected plots
    merger = PdfFileMerger()
    for homedir in folders:
        merger.append(open("{0}/timings/exp0013.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0013.pdf")
