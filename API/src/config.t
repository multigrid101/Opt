c = {}

c._opt_double_precision = false
c._opt_verbosity = 0
c.problemkind = 'gaussNewtonCPU'
c.verboseAD = false

c.backend = require('backend_cuda')

return c
