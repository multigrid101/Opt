#include "main.h"
#include "CombinedSolver.h"
#include "SFSSolverInput.h"
#include "../../shared/ArgParser.h"

int main(int argc, const char * argv[])
{
    std::string inputFilenamePrefix = "../data/shape_from_shading/default";
    /* if (argc >= 2) { */
    /*     inputFilenamePrefix = std::string(argv[1]); */
    /* } */

    bool performanceRun = false;
    /* if (argc > 2) { */
    /*     if (std::string(argv[2]) == "perf") { */
    /*         performanceRun = true; */
    /*     } */
    /*     else { */
    /*         printf("Invalid second parameter: %s\n", argv[2]); */
    /*     } */
    /* } */
    ArgParser argparser;
    argparser.parse(argc, argv);

    int numthreads = argparser.get<int>("numthreads");

    std::string backend = argparser.get<std::string>("backend");

    OptImage::Location location;
    bool onGPU;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
      onGPU = true;
    } else {
      location = OptImage::Location::CPU;
      onGPU = false;
    }

    SFSSolverInput solverInputCPU, solverInputGPU;



    solverInputGPU.load(inputFilenamePrefix, onGPU);
    solverInputGPU.targetDepth->savePLYMesh("sfsInitDepth.ply");

    /* solverInputCPU.load(inputFilenamePrefix, false); */

    CombinedSolverParameters params;
    params.useOpt = true;
    params.nonLinearIter = argparser.get<int>("nIterations");
    params.linearIter = argparser.get<int>("lIterations");
    if (performanceRun) {
        params.useCUDA  = false;
        params.useOpt   = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.nonLinearIter = 60;
        params.linearIter = 10;
    }


    CombinedSolver solver(solverInputGPU, params, location, backend, numthreads);
    /* CombinedSolver solver_cpu(solverInputGPU, params, OptImage::Location::CPU); */

    printf("Solving\n");

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    std::shared_ptr<SimpleBuffer> result = solver.result();
    printf("Solved\n");
    printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
    printf("Save\n");

	return 0;
}
