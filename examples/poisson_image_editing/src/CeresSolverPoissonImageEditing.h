#pragma once
#include "../../shared/CeresSolverBase.h"
#include "../../shared/Config.h"
#include "../../shared/CombinedSolverParameters.h"
class CeresSolverPoissonImageEditing : public CeresSolverBase {
public:
    CeresSolverPoissonImageEditing(const std::vector<unsigned int>& dims, CombinedSolverParameters params) : CeresSolverBase(dims, params) {}
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) override;
};

#if !USE_CERES
double CeresSolverPoissonImageEditing::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, bool profileSolve, std::vector<SolverIteration>& iters) 
{
    return nan("");
}

#endif
