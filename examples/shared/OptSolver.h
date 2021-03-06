#pragma once

#include <cassert>

extern "C" {
#include "Opt.h"
}
#include "OptUtils.h"
#include "CudaArray.h"
#include "SolverIteration.h"
#include "cudaUtil.h"
#include "NamedParameters.h"
#include "SolverBase.h"
#include <cstdlib>
#include <string.h>

/**
 * asdfasdfasdf
 */
static NamedParameters copyParametersAndConvertUnknownsToDouble(const NamedParameters& original) {
    NamedParameters newParams(original);
    std::vector<NamedParameters::Parameter> unknownParameters = original.unknownParameters();
    for (auto p : unknownParameters) {
        auto gpuDoubleImage = copyImageTo(getDoubleImageFromFloatImage(copyImageTo(p.im, OptImage::Location::CPU)), OptImage::Location::GPU);
        newParams.set(p.name, gpuDoubleImage);
    }
    return newParams;
}

static void copyUnknownsFromDoubleToFloat(const NamedParameters& floatParams, const NamedParameters& doubleParams) {
    std::vector<NamedParameters::Parameter> unknownParameters = doubleParams.unknownParameters();
    for (auto p : unknownParameters) {
        auto cpuDoubleImage = copyImageTo(p.im, OptImage::Location::CPU);
        auto cpuFloatImage = getFloatImageFromDoubleImage(cpuDoubleImage);
        NamedParameters::Parameter param;
        floatParams.get(p.name, param);
        copyImage(param.im, cpuFloatImage);
    }
}



class OptSolver : public SolverBase {

public:
    OptSolver(const std::vector<unsigned int>& dimensions, const std::string& terraFile, const std::string& optName, CombinedSolverParameters params, bool doublePrecision = false, std::string backend = "backend_cpu", int numthreads = 1) : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr), m_doublePrecision(doublePrecision)
	{
        Opt_InitializationParameters initParams;
        memset(&initParams, 0, sizeof(Opt_InitializationParameters));
        initParams.verbosityLevel = 1;
        initParams.collectPerKernelTimingInfo = 1;
        initParams.doublePrecision = (int)doublePrecision;


        strcpy(initParams.backend, backend.c_str());


        initParams.numthreads = numthreads;
        initParams.useMaterializedJTJ = params.useMaterializedJTJ;
        initParams.useFusedJTJ = params.useFusedJTJ;


        m_optimizerState = Opt_NewState(initParams);
		m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());
        m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, (unsigned int*)dimensions.data());

		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

    ~OptSolver()
	{
                printf("~OptSolver(): start\n");
		if (m_plan) {
			Opt_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Opt_ProblemDelete(m_optimizerState, m_problem);
		}
                printf("~OptSolver(): stop\n");
	}



    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profiledSolve, std::vector<SolverIteration>& iters) override {

        NamedParameters finalProblemParameters = problemParameters;
        if (m_doublePrecision) {
            finalProblemParameters = copyParametersAndConvertUnknownsToDouble(problemParameters);
        }
        setAllSolverParameters(m_optimizerState, m_plan, solverParameters);
        if (profiledSolve) {
            launchProfiledSolve(m_optimizerState, m_plan, finalProblemParameters.data().data(), iters);
        } else {
            Opt_ProblemSolve(m_optimizerState, m_plan, finalProblemParameters.data().data()); // first data() puts parameters in a std::vector<void*>, second data() returns the underlying pointer to that vector
        }
        m_finalCost = Opt_ProblemCurrentCost(m_optimizerState, m_plan);

        if (m_doublePrecision) {
            copyUnknownsFromDoubleToFloat(problemParameters, finalProblemParameters);
        }

        return m_finalCost;
	}

	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;
    bool m_doublePrecision = false;
};
