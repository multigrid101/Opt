# helper functions that do not fit anywhere else.
from PyPDF2 import PdfFileMerger
import os.path

def mergePdfsOfAvailableFiles(mergedName, folders, formatString):
    """
    mergedName: e.g. ./timings/exp0013.pdf
    folders: e.g. ['arap_mesh_deformation', 'image_warping']
    formatString: e.g. "{0}/timings/exp0013.pdf"
        --> must contain only a single {}
    """
    merger = PdfFileMerger()
    for homedir in folders:
        # first check if file is present, if so then append it
        exp_filename = formatString.format(homedir)
        if os.path.isfile(exp_filename):
            merger.append(open(exp_filename, "rb"))
        else:
            print("mergePdfs(): Warning: no plot-file for {} found".format(homedir))

    merger.write(mergedName)


# paper-sizes in inches
paperSizes = {}
# paperSizes['A4'] = {'height' : 11.69, 'height' : 8.27}
paperSizes['A4'] = (8.27, 11.69)
paperSizes['A5_landscape'] = (8.27, 5.8)
