#pragma once
#include "../../shared/CeresSolverBase.h"
#include "../../shared/Config.h"
#include "../../shared/CombinedSolverParameters.h"
class CeresSolverWarping : public CeresSolverBase {
public:
    CeresSolverWarping(const std::vector<unsigned int>& dims, CombinedSolverParameters params) : CeresSolverBase(dims, params) {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter) override;
};
#if !USE_CERES
double CeresSolverWarping::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iter)
{
    printf("running default method for Ceres solver\n");
    return nan("");
}
#endif
