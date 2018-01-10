from Test import ExampleRun
import re
import pdb

# run all (available) ceres examples just to make sure
# that they compile and don't throw any run-time errors
# USAGE:
# > python test_final_cost.py

# -----------------------------------------------------------------------------
# define list of folders to run:
folders = []

folders.append("arap_mesh_deformation")

folders.append("image_warping")

folders.append("poisson_image_editing")

folders.append("shape_from_shading")

folders.append("volumetric_mesh_deformation")
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# RUN EVERYTHING
tests = []
for homedir in folders:
    print 'begin ' + homedir

    # create, define and run the example
    t = ExampleRun(homedir)
    t._printOutput = False
    t._execCommand = "./" + homedir
    t._args = ["--useCeres", "--useOpt", "false",
        "--nIterations", "5", "--lIterations", "5"]


    t.run()
    print(t._output)

    print 'end ' + homedir
    print ""


# print summary
for t in tests:
    t.printInfo()



