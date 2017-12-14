#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "../../shared/ArgParser.h"
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>

int main(int argc, const char * argv[])
{
    ArgParser argparser;
    argparser.parse(argc, argv);


    std::string filename = "../data/head.ply";
    /* if (argc >= 2) { */
    /*     filename = argv[1]; */
    /* } */
    bool performanceRun = false;
    /* if (argc >= 3) { */
    /*     if (std::string(argv[2]) == "perf") { */
    /*         performanceRun = true; */
    /*     } */
    /*     else { */
    /*         printf("Invalid second parameter: %s\n", argv[2]); */
    /*     } */
    /* } */


    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }

    OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
    int numSubdivides = argparser.get<int>("numSubdivides");
    subdivider.attach(*mesh);
    subdivider(numSubdivides);
    subdivider.detach();


    CombinedSolverParameters params;
    params.useOpt = argparser.get<bool>("useOpt");
    params.useOptLM = argparser.get<bool>("useOptLM");
    params.useCeres = argparser.get<bool>("useCeres"); // makes no sense here

    params.useMaterializedJTJ = argparser.get<bool>("useMaterializedJTJ");
    params.useFusedJTJ = argparser.get<bool>("useMaterializedJTJ");

    /* params.nonLinearIter = 5; //original */
    params.nonLinearIter = argparser.get<int>("nIterations");


    /* params.linearIter = 25; //original */
    params.linearIter = argparser.get<int>("lIterations");

    float weightFit = 1.0f;
    float weightReg = 0.5f;

    int numthreads = argparser.get<int>("numthreads");
    std::string backend = argparser.get<std::string>("backend");

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }

    CombinedSolver solver(mesh, performanceRun, params, weightFit, weightReg, location, backend, numthreads);
    /* CombinedSolver solver_cpu(mesh, performanceRun, params, weightFit, weightReg, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    SimpleMesh* res = solver.result();
    if (!OpenMesh::IO::write_mesh(*res, "out.off"))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << "out.off" << std::endl;
        exit(1);
    }

	return 0;
}
