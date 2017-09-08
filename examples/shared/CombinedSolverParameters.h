#pragma once
#include <string.h>

// TODO this struct mixes compiletime and runtime information --> not good

struct CombinedSolverParameters {
    bool useCUDA = false;
    bool useOpt = true;
    bool useOptLM = false;
    bool useCeres = false;
    bool earlyOut = false;
    unsigned int numIter = 1;
    unsigned int nonLinearIter = 3;
    unsigned int linearIter = 200;
    unsigned int patchIter = 32;
    bool profileSolve = true;
    bool optDoublePrecision = false;
};
