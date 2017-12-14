#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"
#include <string>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>

#include "../../shared/ArgParser.h"
int main(int argc, const char * argv[]) {

    std::string filename = "../data/small_armadillo.ply";

    /* if (argc >= 2) { */
    /*         filename = argv[1]; */
    /* } */


    // For now, any model must be accompanied with a identically 
    // named (besides the extension, which must be 3 characters) mrk file
    std::string markerFilename = filename.substr(0, filename.size() - 3) + "mrk";
    bool performanceRun = false;
    /* if (argc >= 3) { */
    /*     if (std::string(argv[2]) == "perf") { */
    /*         performanceRun = true; */
    /*     } */
    /*     else { */
    /*         printf("Invalid second parameter: %s\n", argv[2]); */
    /*     } */
    /* } */
    int subdivisionFactor = 0;
    bool lmOnlyFullSolve = false;
    /* if (argc > 3) { */
    /*         lmOnlyFullSolve = true; */
    /*         subdivisionFactor = atoi(argv[3]); */
    /* } */

    ArgParser argparser;
    argparser.parse(argc, argv);

    // Load Constraints
    LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename.c_str());

    std::vector<int> constraintsIdx;
    std::vector<std::vector<float>> constraintsTarget;

    for (unsigned int i = 0; i < markersMesh.size(); i++)
    {
            constraintsIdx.push_back(markersMesh[i].getVertexIndex());
            constraintsTarget.push_back(markersMesh[i].getPosition());
    }

    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
    {
            std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
            std::cout << filename << std::endl;
            exit(1);
    }

    OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
    int numSubdivides = argparser.get<int>("numSubdivides");
    if (numSubdivides < 1) {numSubdivides = 1;} // needs to be >=1 for this example or else we get a segfault

    // Initialize subdivider
    if (lmOnlyFullSolve) {
            if (subdivisionFactor > 0) {
                    subdivider.attach(*mesh);
                    subdivider(subdivisionFactor);
                    subdivider.detach();
            }
    } else {
            //OpenMesh::Subdivider::Uniform::CatmullClarkT<SimpleMesh> catmull;
            // Execute 1 subdivision steps
            subdivider.attach(*mesh);
            subdivider(numSubdivides);
            subdivider.detach();
    }
    printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());

    // Load SolverParameter struct
    CombinedSolverParameters params;

    /* params.numIter = 10; */ //original
    /* params.nonLinearIter = 20; */ //original
    /* params.linearIter = 100; */ //original

    params.numIter = argparser.get<int>("oIterations");
    params.nonLinearIter = argparser.get<int>("nIterations");
    params.linearIter = argparser.get<int>("lIterations");

    params.useOpt = argparser.get<bool>("useOpt");
    params.useOptLM = argparser.get<bool>("useOptLM");
    params.useCeres = argparser.get<bool>("useCeres");

    params.useMaterializedJTJ = argparser.get<bool>("useMaterializedJTJ");
    params.useFusedJTJ = argparser.get<bool>("useMaterializedJTJ");

    if (performanceRun) {
        params.useCUDA = false;
        params.useOpt = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.earlyOut = true;
        params.nonLinearIter = 20;
        params.linearIter = 1000;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useOpt = false;
        params.useOptLM = true;
        params.earlyOut = true;
        params.linearIter = 1000;// m_image.getWidth()*m_image.getHeight();
        if (mesh->n_vertices() > 100000) {
            params.nonLinearIter = (unsigned int)mesh->n_vertices() / 5000;
        }
    }

    int numthreads = argparser.get<int>("numthreads");

    std::string backend = argparser.get<std::string>("backend");

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }

    params.optDoublePrecision = false;

    // load scalar problemparameters
    float weightFit = 4.0f;
    float weightReg = 1.0f;

    // Define and call solver
    CombinedSolver solver(mesh, constraintsIdx, constraintsTarget, params, weightFit, weightReg, location, backend, numthreads);
    /* CombinedSolver solver_cpu(mesh, constraintsIdx, constraintsTarget, params, weightFit, weightReg, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    
    // write results to file
    SimpleMesh* res = solver.result();
    if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
    {
            std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
            std::cout << "out.off" << std::endl;
            exit(1);
    }

    return 0;
}
