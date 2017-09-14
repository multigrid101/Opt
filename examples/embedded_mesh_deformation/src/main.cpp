#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"
#include "../../shared/ArgParser.h"

int main(int argc, const char * argv[])
{

  ArgParser argparser;
  argparser.parse(argc, argv);

	std::string filename = "../data/raptor_simplify2k.off";
	/* if (argc >= 2) { */
	/* 	filename = argv[1]; */
	/* } */

    // For now, any model must be accompanied with a identically 
    // named (besides the extension, which must be 3 characters) mrk file
    std::string markerFilename = filename.substr(0, filename.size() - 3) + "mrk";
	// Load Constraints
	LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename.c_str());

	std::vector<int>				constraintsIdx;
	std::vector<std::vector<float>> constraintsTarget;

	for (unsigned int i = 0; i < markersMesh.size(); i++)
	{
        printf("%d: %d\n", i, (markersMesh[i].getVertexIndex()));
		constraintsIdx.push_back(markersMesh[i].getVertexIndex());
		constraintsTarget.push_back(markersMesh[i].getPosition());
	}

	SimpleMesh* mesh = new SimpleMesh();
	if (!OpenMesh::IO::read_mesh(*mesh, filename))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "bunny.off" << std::endl;
		exit(1);
	}
	printf("Beginning MeshDeformationED Warp\n");

    CombinedSolverParameters params;
    /*params.useOpt = true;
    params.useOptLM = false;
    params.numIter = 32;
    params.nonLinearIter = 1;
    params.linearIter = 4000;
    params.earlyOut = false;
    */

    /* LM is good here */
    params.useOpt = false;
    params.useOptLM = true;

    /* params.numIter = 31; // original */
    params.numIter = 1;

    /* params.nonLinearIter = 5; // original */
    params.nonLinearIter = argparser.get<int>("nIterations");

    /* params.linearIter = 125; // original */
    params.linearIter = argparser.get<int>("lIterations");

    std::string backend = argparser.get<std::string>("backend");
    int numthreads = argparser.get<int>("numthreads");

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }

    CombinedSolver solver(mesh, constraintsIdx, constraintsTarget, params, location, backend, numthreads);
    /* CombinedSolver solver_cpu(mesh, constraintsIdx, constraintsTarget, params, OptImage::Location::CPU); */

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
