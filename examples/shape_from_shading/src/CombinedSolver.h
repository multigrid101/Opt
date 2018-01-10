#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"

#include "CUDAImageSolver.h"

#include "CeresImageSolver.h"
#include "SFSSolverInput.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"



class CombinedSolver : public CombinedSolverBase {
private:
    std::shared_ptr<SimpleBuffer>   m_initialUnknown;
    std::shared_ptr<SimpleBuffer>   m_result;
    std::vector<unsigned int> m_dims;
    OptImage::Location m_location;
public:
    CombinedSolver(const SFSSolverInput& inputGPU, CombinedSolverParameters params, OptImage::Location location, std::string backend, int numthreads)
	{
        m_combinedSolverParameters = params;
        m_location = location;

        bool onGPU;
        if (location == OptImage::Location::GPU) {
          onGPU = true;
        } else {
          onGPU = false;
        }

        m_initialUnknown = std::make_shared<SimpleBuffer>(*inputGPU.initialUnknown, onGPU);
        m_result = std::make_shared<SimpleBuffer>(*inputGPU.initialUnknown, onGPU);

        inputGPU.setParameters(m_problemParams, m_result, location);

        m_dims = { (unsigned int)m_result->width(), (unsigned int)m_result->height() };

        addSolver(std::make_shared<CUDAImageSolver>(m_dims), "CUDA", m_combinedSolverParameters.useCUDA);
        addOptSolvers(m_dims, "shape_from_shading.t", m_combinedSolverParameters, m_combinedSolverParameters.optDoublePrecision, backend, numthreads);
        addSolver(std::make_shared<CeresImageSolver>(m_dims, m_combinedSolverParameters), "Ceres", m_combinedSolverParameters.useCeres);
	}

    virtual void combinedSolveInit() override {
        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }

    virtual void preSingleSolve() override {
        resetGPUMemory();
    }
    virtual void postSingleSolve() override {}

    virtual void preNonlinearSolve(int) override {}
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        ceresIterationComparison("Shape From Shading", m_combinedSolverParameters.optDoublePrecision);
    }

    std::shared_ptr<SimpleBuffer> result() {
        return m_result;
    }

	void resetGPUMemory() {
            printf("CombinedSolver::resetGPUMemory(): starting\n");
            if (m_location == OptImage::Location::GPU) {
            printf("calling cudamemcpy\n");
                cudaSafeCall(cudaMemcpy(m_result->data(), m_initialUnknown->data(), m_dims[0]*m_dims[1]*sizeof(float), cudaMemcpyDeviceToDevice));
            } else {
            printf("calling normal memcpy\n");
                memcpy(m_result->data(), m_initialUnknown->data(), m_dims[0]*m_dims[1]*sizeof(float));
            }
            printf("CombinedSolver::resetGPUMemory(): stopping\n");
	}

};
