local c = {}

c._opt_double_precision = false
-- c._opt_double_precision = true
if c._opt_double_precision then c.opt_float =  double else c.opt_float =  float end

c._opt_verbosity = 10
c.problemkind = 'gaussNewtonCPU'
c.verboseAD = false

-- c.numthreads = 2 -- only for backend_cpu_mt, no effect on other backends
-- c.numthreads = 4 -- only for backend_cpu_mt, no effect on other backends
c.numthreads = _opt_numthreads -- only for backend_cpu_mt, no effect on other backends

-- c.cpumap = { 0, 1, 2, 3, 4, 5, 6, 7 } -- default -- TODO this feature is broken at the moment
-- c.cpumap = { 1, 5, 2, 3, 4, 5, 6, 7 }

-- NOTE: The size of a single mutex is TODO remove this
-- c.nummutexes = 10000 -- adjust by hand (only has effec for backend_cpu_mt) TODO find better solution for this
-- c.nummutexes = 386 -- adjust by hand (only has effec for backend_cpu_mt) TODO find better solution for this

c.backend = _opt_backend -- configured in cpp code

-- c.use_contiguous_allocation = false
c.use_contiguous_allocation = true

-- c.use_bindless_texture = true
c.use_bindless_texture = false

c.GRID_SIZES = { {256,1,1}, {16,16,1}, {8,8,4} } -- only relevant for CUDA


-- c.pascalOrBetterGPU = true
c.pascalOrBetterGPU = false

return c
