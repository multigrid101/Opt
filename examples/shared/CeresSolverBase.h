#pragma once

#include "SolverBase.h"
#include "Config.h"
#include "CombinedSolverParameters.h"

#if USE_CERES
#define GLOG_NO_ABBREVIATED_SEVERITIES
/* #include "ceres/ceres.h" */
#include "/home/sebastian/Desktop/ceres-solver-1.13.0/include/ceres/ceres.h"
#include "glog/logging.h"
using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
#endif
#include <memory>
class CeresSolverBase : public SolverBase {
public:
    CeresSolverBase(const std::vector<unsigned int>& dims, CombinedSolverParameters params) : m_dims(dims), m_params(params) {}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override {
        fprintf(stderr, "No Ceres solve implemented\n");
        return m_finalCost;
    }

protected:
#if USE_CERES
    double launchProfiledSolveAndSummary(const std::unique_ptr<Solver::Options>& options, Problem* problem, bool profileSolve, std::vector<SolverIteration>& iter);
    std::unique_ptr<Solver::Options> initializeOptions(const NamedParameters& solverParameters) const;
#endif
    std::vector<unsigned int> m_dims;
    CombinedSolverParameters m_params;
};
