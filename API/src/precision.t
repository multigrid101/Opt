conf = require('config')
-- Switch to double to check for precision issues in the solver
-- using double incurs bandwidth, compute, and atomic performance penalties
if conf._opt_double_precision then
	opt_float =  conf.opt_float
else
	opt_float =  conf.opt_float
end
