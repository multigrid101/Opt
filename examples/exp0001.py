from myPlots import ceresVsOptCpuBar
from myTimings import doTimingsCeresVsOptCpu
from myArgParsers import experimentParser
from PyPDF2 import PdfFileMerger
import sys


folders = []
folders.append("arap_mesh_deformation")

folders.append("image_warping")

folders.append("poisson_image_editing")

folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")


exParser = experimentParser()
exParser.parse_args(len(sys.argv), sys.argv)



if exParser._doTimeAll:
    for homedir in folders:
        print("exp0001.py: performing simulations for {0}".format(homedir))
        doTimingsCeresVsOptCpu(homedir)

if exParser._doPlotAll:
    for homedir in folders:
        print("exp0001.py: creating plot for {0}".format(homedir))
        ceresVsOptCpuBar(homedir)

    merger = PdfFileMerger()
    for homedir in folders:
        # then append it to the Merger
        merger.append(open("{0}/timings/ceresVsOptCpu.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0001.pdf")

if exParser._exampleName is not None:
    homedir = exParser._exampleName

    print("exp0001.py: performing simulations for {0}".format(homedir))
    doTimingsCeresVsOptCpu(exParser._exampleName)

    print("exp0001.py: creating plot for {0}".format(homedir))
    ceresVsOptCpuBar(exParser._exampleName)

    # re-do the file with the collected plots
    merger = PdfFileMerger()
    for homedir in folders:
        merger.append(open("{0}/timings/ceresVsOptCpu.pdf".format(homedir), "rb"))
    merger.write("./timings/exp0001.pdf")
