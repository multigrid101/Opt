local c = {}

c._opt_double_precision = false
-- c._opt_double_precision = true
if c._opt_double_precision then c.opt_float =  double else c.opt_float =  float end

c._opt_verbosity = 10
c.problemkind = 'gaussNewtonCPU'
c.verboseAD = false

-- c.numthreads = 2 -- only for backend_cpu_mt, no effect on other backends
c.numthreads = 4 -- only for backend_cpu_mt, no effect on other backends
-- c.cpumap = { 0, 1, 2, 3, 4, 5, 6, 7 } -- default
-- c.cpumap = { 1, 5, 2, 3, 4, 5, 6, 7 }
c.nummutexes = 10000 -- adjust by hand (only has effec for backend_cpu_mt) TODO find better solution for this

-- c.backend = 'backend_cuda'
c.backend = 'backend_cpu'
-- c.backend = 'backend_cpu_mt'

c.use_contiguous_allocation = false
-- c.use_bindless_texture = true
c.use_bindless_texture = false

c.GRID_SIZES = { {256,1,1}, {16,16,1}, {8,8,4} } -- only relevant for CUDA


c.pascalOrBetterGPU = false

return c
