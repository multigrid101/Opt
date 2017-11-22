#include "mLibInclude.h"
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
	std::string filename = "../data/head.ply";
	/* if (argc >= 2) { */
	/* 	filename = argv[1]; */
	/* } */
        ArgParser argparser;
        argparser.parse(argc, argv);

	SimpleMesh* mesh = new SimpleMesh();

	if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}
	printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());

    CombinedSolverParameters params;

    /* params.nonLinearIter = 20;//original */
    params.nonLinearIter = argparser.get<int>("nIterations");

    /* params.linearIter = 60;//original */
    params.linearIter = argparser.get<int>("lIterations");

    OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
    int numSubdivides = argparser.get<int>("numSubdivides");
    subdivider.attach(*mesh);
    subdivider(numSubdivides);
    subdivider.detach();

    int3 voxelGridSize = make_int3(5*(numSubdivides+1), 20*(numSubdivides*1), 5*(numSubdivides+1));

    int numthreads = argparser.get<int>("numthreads");
    std::string backend = argparser.get<std::string>("backend");

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }

    CombinedSolver solver(mesh, voxelGridSize, params, location, backend, numthreads);
    /* CombinedSolver solver_cpu(mesh, voxelGridSize, params, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    SimpleMesh* res = solver.result();
    solver.saveGraphResults();

	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	return 0;
}
