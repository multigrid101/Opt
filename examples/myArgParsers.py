import sys

# A collection of custom argument parsers, usually used in many files so I collect them here

# When reading the code, remember that sys.argv always contains the name of the
# script, so argc==1 means that NO extra arguments were passed.

validExampleNames = ["arap_mesh_deformation",
        "cotangent_mesh_smoothing",
        "embedded_mesh_deformation",
        "image_warping",
        "intrincsic_image_decomposition",
        "optical_flow",
        "poisson_image_editing",
        "robust_nonrigid_alignment",
        "shape_from_shading",
        "volumentric_mesh_deformation"]

class experimentParser:
    def __init__(self):
        self._helptext = """
                         simple parser for experiments. possibilities for the arglist:
                         "time" --> performs timing for all examples
                         "plot" --> creates plots for all examples
                         "time plot" --> does both
                         "image_warping" --> both, but only for one examples
                         """
        self._doTimeAll = False
        self._doPlotAll = False
        self._exampleName = None


    def parse_args(self, argc, argv):
        # if there are no args, we do everything
        if argc == 1:
            self._doTimeAll = True
            self._doPlotAll = True
            return

        # if there is one arg, its either 'time', 'plot' or an example name
        if argc == 2:
            if argv[1] == "time":
                self._doTimeAll = True
                return

            if argv[1] == "plot":
                self._doPlotAll = True
                return

            if argv[1] in validExampleNames:
                self._exampleName = argv[1]
                return

        # if there are two args, then they must be "time plot" in that
        # or the other order
        if argc == 3:
            if "time" in argv and "plot" in argv:
                self._doTimeAll = True
                self._doPlotAll = True
                return


        # anything else is an error
        errMsg = """
        Invalid argument List:

        {0}


        """.format(str(argv))
        sys.exit(errMsg)
