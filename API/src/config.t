c = {}

c._opt_double_precision = false
c._opt_verbosity = 0
c.problemkind = 'gaussNewtonCPU'
c.verboseAD = false

c.backend = 'backend_cuda'

c.GRID_SIZES = { {256,1,1}, {16,16,1}, {8,8,4} } -- only relevant for CUDA

return c
