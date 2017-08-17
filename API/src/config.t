local c = {}

c._opt_double_precision = false
if c._opt_double_precision then c.opt_float =  double else c.opt_float =  float end

c._opt_verbosity = 0
c.problemkind = 'gaussNewtonCPU'
c.verboseAD = false

c.numthreads = 2 -- only for backend_cpu_mt

c.backend = 'backend_cuda'
-- c.backend = 'backend_cpu'
-- c.backend = 'backend_cpu_mt'

c.use_contiguous_allocation = false
c.use_bindless_texture = true
-- c.use_bindless_texture = false

c.GRID_SIZES = { {256,1,1}, {16,16,1}, {8,8,4} } -- only relevant for CUDA


c.pascalOrBetterGPU = false

return c
