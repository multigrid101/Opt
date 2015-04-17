
local S = require("std")
local util = require("util")
local C = util.C

solversCPU = {}

solversCPU.gradientDescentCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		gradient : vars.unknownType
		dimensions : int64[#vars.dimensions + 1]
	}

	local computeCost = util.makeComputeCost(tbl, vars.imagesAll)
	local computeGradient = util.makeComputeGradient(tbl, vars.unknownType, vars.imagesAll)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dimensions)]

		-- TODO: parameterize these
		var initialLearningRate = 0.01
		var maxIters = 200
		var tolerance = 1e-10

		-- Fixed constants (these do not need to be parameterized)
		var learningLoss = 0.8
		var learningGain = 1.1
		var minLearningRate = 1e-25

		var learningRate = initialLearningRate

		for iter = 0, maxIters do
			var startCost = computeCost(vars.imagesAll)
			solverLog("iteration %d, cost=%f, learningRate=%f\n", iter, startCost, learningRate)
			--C.getchar()

			computeGradient(pd.gradient, vars.imagesAll)
			
			--
			-- move along the gradient by learningRate
			--
			var maxDelta = 0.0
			for h = 0,pd.gradH do
				for w = 0,pd.gradW do
					var addr = &vars.unknownImage(w, h)
					var delta = learningRate * pd.gradient(w, h)
					@addr = @addr - delta
					maxDelta = util.max(C.fabsf(delta), maxDelta)
				end
			end

			--
			-- update the learningRate
			--
			var endCost = computeCost(vars.imagesAll)
			if endCost < startCost then
				learningRate = learningRate * learningGain

				if maxDelta < tolerance then
					solverLog("terminating, maxDelta=%f\n", maxDelta)
					break
				end
			else
				learningRate = learningRate * learningLoss

				if learningRate < minLearningRate then
					solverLog("terminating, learningRate=%f\n", learningRate)
					break
				end
			end
		end
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]

		pd.gradient:initCPU(pd.gradW, pd.gradH)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.conjugateGradientCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		costW : int
		costH : int
		dimensions : int64[#vars.dimensions + 1]
				
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType
		
		searchDirection : vars.unknownType
	}
	
	local cpu = util.makeCPUFunctions(tbl, vars.unknownType, vars.dataImages, vars.imagesAll)
	
	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dimensions)]
		
		var maxIters = 1000
		
		var prevBestAlpha = 0.0

		for iter = 0, maxIters do

			var iterStartCost = cpu.computeCost(vars.imagesAll)
			solverLog("iteration %d, cost=%f\n", iter, iterStartCost)

			cpu.computeGradient(pd.gradient, vars.imagesAll)
			
			--
			-- compute the search direction
			--
			var beta = 0.0
			if iter == 0 then
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						pd.searchDirection(w, h) = -pd.gradient(w, h)
					end
				end
			else
				var num = 0.0
				var den = 0.0
				
				--
				-- Polak-Ribiere conjugacy
				-- 
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						var g = pd.gradient(w, h)
						var p = pd.prevGradient(w, h)
						num = num + (-g * (-g + p))
						den = den + p * p
					end
				end
				beta = util.max(num / den, 0.0)
				
				var epsilon = 1e-5
				if den > -epsilon and den < epsilon then
					beta = 0.0
				end
				
				for h = 0, pd.gradH do
					for w = 0, pd.gradW do
						pd.searchDirection(w, h) = -pd.gradient(w, h) + beta * pd.searchDirection(w, h)
					end
				end
			end
			
			C.memcpy(pd.prevGradient.impl.data, pd.gradient.impl.data, sizeof(float) * pd.gradW * pd.gradH)
			C.memcpy(pd.currentValues.impl.data, vars.unknownImage.impl.data, sizeof(float) * pd.gradW * pd.gradH)
			
			--
			-- line search
			--
			cpu.computeResiduals(pd.currentValues, pd.currentResiduals, vars.dataImages)
			
			-- NOTE: this approach to line search will have unexpected behavior if the cost function
			-- returns double-precision, but residuals are stored at single precision!
			
			var bestAlpha = 0.0
			
			var useBruteForce = (iter <= 1) or prevBestAlpha == 0.0
			if not useBruteForce then
				
				bestAlpha = cpu.lineSearchQuadraticMinimum(pd.currentValues, pd.currentResiduals, pd.searchDirection, vars.unknownImage, prevBestAlpha, vars.dataImages)
				
				if bestAlpha == 0.0 then useBruteForce = true end
			end
			
			if useBruteForce then
				solverLog("brute-force line search\n")
				bestAlpha = cpu.lineSearchBruteForce(pd.currentValues, pd.currentResiduals, pd.searchDirection, vars.unknownImage, vars.dataImages)
			end
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					vars.unknownImage(w, h) = pd.currentValues(w, h) + bestAlpha * pd.searchDirection(w, h)
				end
			end
			
			prevBestAlpha = bestAlpha
			
			--if iter % 20 == 0 then C.getchar() end
			
			solverLog("alpha=%12.12f, beta=%12.12f\n\n", bestAlpha, beta)
			if bestAlpha == 0.0 and beta == 0.0 then
				break
			end
		end
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]
		
		pd.costW = pd.dimensions[vars.costWIndex]
		pd.costH = pd.dimensions[vars.costHIndex]

		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		
		pd.currentResiduals:initCPU(pd.costW, pd.costH)
		
		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.searchDirection:initCPU(pd.gradW, pd.gradH)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.linearizedConjugateGradientCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dimensions : int64[#vars.dimensions + 1]
		
		b : vars.unknownType
		r : vars.unknownType
		p : vars.unknownType
		zeroes : vars.unknownType
		Ap : vars.unknownType
	}
	
	local computeCost = util.makeComputeCost(tbl, vars.imagesAll)
	local imageInnerProduct = util.makeImageInnerProduct(vars.unknownType)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dimensions)]
		
		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				pd.r(w, h) = -tbl.gradient.boundary(w, h, vars.unknownImage, vars.dataImages)
				pd.b(w, h) = tbl.gradient.boundary(w, h, pd.zeroes, vars.dataImages)
				pd.p(w, h) = pd.r(w, h)
			end
		end
		
		var rTr = imageInnerProduct(pd.r, pd.r)

		for iter = 0,maxIters do

			var iterStartCost = computeCost(vars.imagesAll)
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.Ap(w, h) = tbl.gradient.boundary(w, h, pd.p, vars.dataImages) - pd.b(w, h)
				end
			end
			
			var den = imageInnerProduct(pd.p, pd.Ap)
			var alpha = rTr / den
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					vars.unknownImage(w, h) = vars.unknownImage(w, h) + alpha * pd.p(w, h)
					pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
				end
			end
			
			var rTrNew = imageInnerProduct(pd.r, pd.r)
			
			solverLog("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTrNew)
			
			if(rTrNew < tolerance) then break end
			
			var beta = rTrNew / rTr
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.p(w, h) = pd.r(w, h) + beta * pd.p(w, h)
				end
			end
			
			rTr = rTrNew
		end
		
		var finalCost = computeCost(vars.imagesAll)
		solverLog("final cost=%f\n", finalCost)
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]

		pd.b:initCPU(pd.gradW, pd.gradH)
		pd.r:initCPU(pd.gradW, pd.gradH)
		pd.p:initCPU(pd.gradW, pd.gradH)
		pd.Ap:initCPU(pd.gradW, pd.gradH)
		pd.zeroes:initCPU(pd.gradW, pd.gradH)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.linearizedPreconditionedConjugateGradientCPU = function(Problem, tbl, vars)
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		dimensions : int64[#vars.dimensions + 1]
		
		b : vars.unknownType
		r : vars.unknownType
		z : vars.unknownType
		p : vars.unknownType
		MInv : vars.unknownType
		Ap : vars.unknownType
		zeroes : vars.unknownType
	}
	
	local computeCost = util.makeComputeCost(tbl, vars.imagesAll)
	local imageInnerProduct = util.makeImageInnerProduct(vars.unknownType)

	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dimensions)]

		-- TODO: parameterize these
		var maxIters = 1000
		var tolerance = 1e-5

		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				--pd.MInv(w, h) = 1.0 / tbl.gradientPreconditioner(w, h)
				pd.MInv(w, h) = 1.0
			end
		end
		
		for h = 0, pd.gradH do
			for w = 0, pd.gradW do
				pd.r(w, h) = -tbl.gradient.boundary(w, h, vars.unknownImage, vars.dataImages)
				pd.b(w, h) = tbl.gradient.boundary(w, h, pd.zeroes, vars.dataImages)
				pd.z(w, h) = pd.MInv(w, h) * pd.r(w, h)
				pd.p(w, h) = pd.z(w, h)
			end
		end
		
		for iter = 0,maxIters do

			var iterStartCost = computeCost(vars.imagesAll)
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.Ap(w, h) = tbl.gradient.boundary(w, h, pd.p, vars.dataImages) - pd.b(w, h)
				end
			end
			
			var rTzStart = imageInnerProduct(pd.r, pd.z)
			var den = imageInnerProduct(pd.p, pd.Ap)
			var alpha = rTzStart / den
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					vars.unknownImage(w, h) = vars.unknownImage(w, h) + alpha * pd.p(w, h)
					pd.r(w, h) = pd.r(w, h) - alpha * pd.Ap(w, h)
				end
			end
			
			var rTr = imageInnerProduct(pd.r, pd.r)
			
			solverLog("iteration %d, cost=%f, rTr=%f\n", iter, iterStartCost, rTr)
			
			if(rTr < tolerance) then break end
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.z(w, h) = pd.MInv(w, h) * pd.r(w, h)
				end
			end
			
			var beta = imageInnerProduct(pd.z, pd.r) / rTzStart
			
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.p(w, h) = pd.z(w, h) + beta * pd.p(w, h)
				end
			end
		end
		
		var finalCost = computeCost(vars.imagesAll)
		solverLog("final cost=%f\n", finalCost)
	end

	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]

		pd.b:initCPU(pd.gradW, pd.gradH)
		pd.r:initCPU(pd.gradW, pd.gradH)
		pd.z:initCPU(pd.gradW, pd.gradH)
		pd.p:initCPU(pd.gradW, pd.gradH)
		pd.MInv:initCPU(pd.gradW, pd.gradH)
		pd.Ap:initCPU(pd.gradW, pd.gradH)
		pd.zeroes:initCPU(pd.gradW, pd.gradH)
		
		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

solversCPU.lbfgsCPU = function(Problem, tbl, vars)

	local maxIters = 1000
	
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		costW : int
		costH : int
		dimensions : int64[#vars.dimensions + 1]
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType
				
		p : vars.unknownType
		sList : vars.unknownType[maxIters]
		yList : vars.unknownType[maxIters]
		syProduct : float[maxIters]
		yyProduct : float[maxIters]
		alphaList : float[maxIters]
		
		-- variables used for line search
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
	}
	
	local cpu = util.makeCPUFunctions(tbl, vars.unknownType, vars.dataImages, vars.imagesAll)
	
	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)

		-- two-loop recursion: http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf
		
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dimensions)]

		var m = 10
		var k = 0
		
		var prevBestAlpha = 0.0
		
		cpu.computeGradient(pd.gradient, vars.imagesAll)

		for iter = 0, maxIters - 1 do

			var iterStartCost = cpu.computeCost(vars.imagesAll)
			solverLog("iteration %d, cost=%f\n", iter, iterStartCost)
			
			--
			-- compute the search direction p
			--
			cpu.setImage(pd.p, pd.gradient, -1.0f)
			
			if k >= 1 then
				for i = k - 1, k - m - 1, -1 do
					if i < 0 then break end
					pd.alphaList[i] = cpu.imageInnerProduct(pd.sList[i], pd.p) / pd.syProduct[i]
					cpu.addImage(pd.p, pd.yList[i], -pd.alphaList[i])
				end
				var scale = pd.syProduct[k - 1] / pd.yyProduct[k - 1]
				cpu.scaleImage(pd.p, scale)
				for i = k - m, k do
					if i >= 0 then
						var beta = cpu.imageInnerProduct(pd.yList[i], pd.p) / pd.syProduct[i]
						cpu.addImage(pd.p, pd.sList[i], pd.alphaList[i] - beta)
					end
				end
			end
			
			--
			-- line search
			--
			cpu.copyImage(pd.currentValues, vars.unknownImage)
			cpu.computeResiduals(pd.currentValues, pd.currentResiduals, vars.dataImages)
			
			var bestAlpha = 0.0
			
			var useBruteForce = (iter <= 1) or prevBestAlpha == 0.0
			if not useBruteForce then
				
				bestAlpha = cpu.lineSearchQuadraticMinimum(pd.currentValues, pd.currentResiduals, pd.p, vars.unknownImage, prevBestAlpha, vars.dataImages)
				
				if bestAlpha == 0.0 then
					solverLog("quadratic guess=%f failed, trying again...\n", prevBestAlpha)
					bestAlpha = cpu.lineSearchQuadraticMinimum(pd.currentValues, pd.currentResiduals, pd.p, vars.unknownImage, prevBestAlpha * 4.0, vars.dataImages)
					
					if bestAlpha == 0.0 then
					
						if iter >= 10 then
							solverLog("quadratic minimization exhausted\n")
						else
							useBruteForce = true
						end
						--cpu.dumpLineSearch(pd.currentValues, pd.currentResiduals, pd.p, vars.unknownImage, vars.dataImages)
					end
				end
			end
			
			if useBruteForce then
				solverLog("brute-force line search\n")
				bestAlpha = cpu.lineSearchBruteForce(pd.currentValues, pd.currentResiduals, pd.p, vars.unknownImage, vars.dataImages)
			end
			
			-- compute new x and s
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					var delta = bestAlpha * pd.p(w, h)
					vars.unknownImage(w, h) = pd.currentValues(w, h) + delta
					pd.sList[k](w, h) = delta
				end
			end
			
			cpu.copyImage(pd.prevGradient, pd.gradient)
			
			cpu.computeGradient(pd.gradient, vars.imagesAll)
			
			-- compute new y
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.yList[k](w, h) = pd.gradient(w, h) - pd.prevGradient(w, h)
				end
			end
			
			pd.syProduct[k] = cpu.imageInnerProduct(pd.sList[k], pd.yList[k])
			pd.yyProduct[k] = cpu.imageInnerProduct(pd.yList[k], pd.yList[k])
			
			prevBestAlpha = bestAlpha
			
			k = k + 1
			
			solverLog("alpha=%12.12f\n\n", bestAlpha)
			if bestAlpha == 0.0 then
				break
			end
		end
	end
	
	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i+1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]
		
		pd.costW = pd.dimensions[vars.costWIndex]
		pd.costH = pd.dimensions[vars.costHIndex]

		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		pd.currentResiduals:initCPU(pd.costW, pd.costH)
		
		pd.p:initCPU(pd.gradW, pd.gradH)
		
		for i = 0, maxIters - 1 do
			pd.sList[i]:initCPU(pd.gradW, pd.gradH)
			pd.yList[i]:initCPU(pd.gradW, pd.gradH)
		end

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

-- vector-free L-BFGS using two-loop recursion: http://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf
solversCPU.vlbfgsCPU = function(Problem, tbl, vars)

	local maxIters = 1000
	local m = 4
	local b = 2 * m + 1
	
	local struct PlanData(S.Object) {
		plan : opt.Plan
		gradW : int
		gradH : int
		costW : int
		costH : int
		dimensions : int64[#vars.dimensions + 1]
		
		gradient : vars.unknownType
		prevGradient : vars.unknownType

		p : vars.unknownType
		sList : vars.unknownType[m]
		yList : vars.unknownType[m]
		alphaList : float[maxIters]
		
		dotProductMatrix : vars.unknownType
		dotProductMatrixStorage : vars.unknownType
		coefficients : double[b]
		
		-- variables used for line search
		currentValues : vars.unknownType
		currentResiduals : vars.unknownType
	}
	
	local terra imageFromIndex(pd : &PlanData, index : int)
		if index < m then
			return pd.sList[index]
		elseif index < 2 * m then
			return pd.yList[index - m]
		else
			return pd.gradient
		end
	end
	
	local terra nextCoefficientIndex(index : int)
		if index == m - 1 or index == 2 * m - 1 or index == 2 * m then
			return -1
		end
		return index + 1
	end
	
	local cpu = util.makeCPUFunctions(tbl, vars.unknownType, vars.dataImages, vars.imagesAll)
	
	local terra impl(data_ : &opaque, imageBindings : &&opt.ImageBinding, params_ : &opaque)
		
		var pd = [&PlanData](data_)
		var params = [&double](params_)
		var dimensions = pd.dimensions

		var [vars.imagesAll] = [util.getImages(vars, imageBindings, dimensions)]

		var k = 0
		
		-- using an initial guess of alpha means that it will invoke quadratic optimization on the first iteration,
		-- which is only sometimes a good idea.
		var prevBestAlpha = 1.0
		
		cpu.computeGradient(pd.gradient, vars.imagesAll)

		for iter = 0, maxIters - 1 do

			var iterStartCost = cpu.computeCost(vars.imagesAll)
			solverLog("iteration %d, cost=%f\n", iter, iterStartCost)
			
			--
			-- compute the search direction p
			--
			if k == 0 then
				cpu.setImage(pd.p, pd.gradient, -1.0f)
			else
				-- compute the top half of the dot product matrix
				cpu.copyImage(pd.dotProductMatrixStorage, pd.dotProductMatrix)
				for i = 0, b do
					for j = i, b do
						var prevI = nextCoefficientIndex(i)
						var prevJ = nextCoefficientIndex(j)
						if prevI == -1 or prevJ == -1 then
							pd.dotProductMatrix(i, j) = cpu.imageInnerProduct(imageFromIndex(pd, i), imageFromIndex(pd, j))
						else
							pd.dotProductMatrix(i, j) = pd.dotProductMatrixStorage(prevI, prevJ)
						end
					end
				end
				
				-- compute the bottom half of the dot product matrix
				for i = 0, b do
					for j = 0, i do
						pd.dotProductMatrix(i, j) = pd.dotProductMatrix(j, i)
					end
				end
			
				for i = 0, 2 * m do pd.coefficients[i] = 0.0 end
				pd.coefficients[2 * m] = -1.0
				
				for i = k - 1, k - m - 1, -1 do
					if i < 0 then break end
					var j = i - (k - m)
					
					var num = 0.0
					for q = 0, b do
						num = num + pd.coefficients[q] * pd.dotProductMatrix(q, j)
					end
					var den = pd.dotProductMatrix(j, j + m)
					pd.alphaList[i] = num / den
					pd.coefficients[j + m] = pd.coefficients[j + m] - pd.alphaList[i]
				end
				
				var scale = pd.dotProductMatrix(m - 1, 2 * m - 1) / pd.dotProductMatrix(2 * m - 1, 2 * m - 1)
				for i = 0, b do
					pd.coefficients[i] = pd.coefficients[i] * scale
				end
				
				for i = k - m, k do
					if i >= 0 then
						var j = i - (k - m)
						var num = 0.0
						for q = 0, b do
							num = num + pd.coefficients[q] * pd.dotProductMatrix(q, m + j)
						end
						var den = pd.dotProductMatrix(j, j + m)
						var beta = num / den
						pd.coefficients[j] = pd.coefficients[j] + (pd.alphaList[i] - beta)
					end
				end
				
				--
				-- reconstruct p from basis vectors
				--
				cpu.scaleImage(pd.p, 0.0f)
				for i = 0, b do
					var image = imageFromIndex(pd, i)
					var coefficient = pd.coefficients[i]
					cpu.addImage(pd.p, image, coefficient)
				end
			end
			
			--
			-- line search
			--
			cpu.copyImage(pd.currentValues, vars.unknownImage)
			cpu.computeResiduals(pd.currentValues, pd.currentResiduals, vars.dataImages)
			
			var bestAlpha = cpu.lineSearchQuadraticFallback(pd.currentValues, pd.currentResiduals, pd.p, vars.unknownImage, prevBestAlpha, vars.dataImages)
			
			-- remove the oldest s and y
			for i = 0, m - 1 do
				pd.yList[i] = pd.yList[i + 1]
				pd.sList[i] = pd.sList[i + 1]
			end
			
			-- compute new x and s
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					var delta = bestAlpha * pd.p(w, h)
					vars.unknownImage(w, h) = pd.currentValues(w, h) + delta
					pd.sList[m - 1](w, h) = delta
				end
			end
			
			cpu.copyImage(pd.prevGradient, pd.gradient)
			
			cpu.computeGradient(pd.gradient, vars.imagesAll)
			
			-- compute new y
			for h = 0, pd.gradH do
				for w = 0, pd.gradW do
					pd.yList[m - 1](w, h) = pd.gradient(w, h) - pd.prevGradient(w, h)
				end
			end
			
			prevBestAlpha = bestAlpha
			
			k = k + 1
			
			solverLog("alpha=%12.12f\n\n", bestAlpha)
			if bestAlpha == 0.0 then
				break
			end
		end
	end
	
	local terra makePlan(actualDims : &uint64) : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.impl = impl
		pd.dimensions[0] = 1
		for i = 0,[#vars.dimensions] do
			pd.dimensions[i + 1] = actualDims[i]
		end

		pd.gradW = pd.dimensions[vars.gradWIndex]
		pd.gradH = pd.dimensions[vars.gradHIndex]
		
		pd.costW = pd.dimensions[vars.costWIndex]
		pd.costH = pd.dimensions[vars.costHIndex]

		pd.gradient:initCPU(pd.gradW, pd.gradH)
		pd.prevGradient:initCPU(pd.gradW, pd.gradH)
		
		pd.currentValues:initCPU(pd.gradW, pd.gradH)
		pd.currentResiduals:initCPU(pd.costW, pd.costH)
		
		pd.p:initCPU(pd.gradW, pd.gradH)
		
		for i = 0, m do
			pd.sList[i]:initCPU(pd.gradW, pd.gradH)
			pd.yList[i]:initCPU(pd.gradW, pd.gradH)
		end
		
		pd.dotProductMatrix:initCPU(b, b)
		pd.dotProductMatrixStorage:initCPU(b, b)

		return &pd.plan
	end
	return Problem:new { makePlan = makePlan }
end

return solversCPU