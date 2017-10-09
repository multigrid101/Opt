from Test import Test
from Test import TestGroup

# list of folders to run:


folders = []
folders.append("arap_mesh_deformation")
folders.append("cotangent_mesh_smoothing")
folders.append("embedded_mesh_deformation")
folders.append("image_warping")
folders.append("intrinsic_image_decomposition")
folders.append("poisson_image_editing")
folders.append("robust_nonrigid_alignment") # note: this solver does more than one solve because singleSolve is overwritten by CombinedSolver
folders.append("volumetric_mesh_deformation")


# reference costs for original problem size and (nIter, nNonlinIter, nLinIter) = (1,1,1)
# reference cost should always be taken from CUDA!!!
referenceCosts = {}
referenceCosts['arap_mesh_deformation'] = 5867.8574 # CUDA
referenceCosts['cotangent_mesh_smoothing'] = 10534.742 # CUDA
referenceCosts['embedded_mesh_deformation'] = 0.36712926 # CUDA
referenceCosts['image_warping'] = 1774.3405 # CUDA
referenceCosts['intrinsic_image_decomposition'] = 26014334.0 # CUDA
referenceCosts['poisson_image_editing'] = 15808011.0 # CUDA
referenceCosts['robust_nonrigid_alignment'] = 66.784683 # CUDA cost of first!!! solve
referenceCosts['volumetric_mesh_deformation'] = 189.74081 # CUDA

backends = ["backend_cuda", "backend_cpu", "backend_cpu_mt"]
# backends = ["backend_cpu", "backend_cpu_mt"]
numthreads = ["1", "2", "4", "8"]
# numthreads = ["1", "2", "4", "8", "16", "32"]
# numthreads = ["4", "8"]



for homedir in folders:
    tests = TestGroup()
    print 'begin ' + homedir
    for backend in backends:
        for num in numthreads:
            test = Test(homedir)
            test.setBackend(backend)
            test.setNumthreads(num)
            test.setTolerance(1e-6)
            test.setReferenceCost(referenceCosts[homedir])
            test._checkResult = True
            test._printOutput = True
            tests.addTest(test)

    tests.runAll()
    print 'end homedir'

