#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "../../shared/ArgParser.h"
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>

static SimpleMesh* createMesh(std::string filename) {
    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());
    return mesh;
}

static std::vector<int4> getSourceTetIndices(std::string filename) {
    // TODO: error handling
    std::ifstream inFile(filename);
    int tetCount = 0;
    int temp;
    inFile >> tetCount >> temp >> temp;
    std::vector<int4> tets(tetCount);
    for (int i = 0; i < tetCount; ++i) {
        inFile >> temp >> tets[i].x >> tets[i].y >> tets[i].z >> tets[i].w;
    }
    int4 f = tets[tets.size() - 1];
    printf("Final tet read: %d %d %d %d\n", f.x, f.y, f.z, f.w);
    return tets;
}

int main(int argc, const char * argv[])
{
    std::string targetSourceDirectory = "../data/squat_target";
    std::string sourceFilename = "../data/squat_source.obj";
    std::string tetmeshFilename = "../data/squat_tetmesh.ele";

    ArgParser argparser;
    argparser.parse(argc, argv);

    /* if (argc > 1) { */
    /*     assert(argc > 3); */
    /*     targetSourceDirectory = argv[1]; */
    /*     sourceFilename = argv[2]; */
    /*     tetmeshFilename = argv[3]; */
    /* } */

    std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);
    

    std::vector<int4> sourceTetIndices = getSourceTetIndices(tetmeshFilename);

    SimpleMesh* sourceMesh = createMesh(sourceFilename);

    OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
    int numSubdivides = argparser.get<int>("numSubdivides");
    subdivider.attach(*sourceMesh);
    subdivider(numSubdivides);
    subdivider.detach();

    std::vector<SimpleMesh*> targetMeshes;
    for (auto target : targetFiles) {
        targetMeshes.push_back(createMesh(targetSourceDirectory + "/" + target));
    }
    std::cout << "All meshes now in memory" << std::endl;

    CombinedSolverParameters params;

    /* params.numIter = 15;//original */
    params.numIter = 1;

    /* params.nonLinearIter = 10;//original */
    params.nonLinearIter = argparser.get<int>("nIterations");

    /* params.linearIter = 250;//original */
    params.linearIter = argparser.get<int>("lIterations");

    params.useOpt = false;
    params.useOptLM = true;

    int numthreads = argparser.get<int>("numthreads");
    std::string backend = argparser.get<std::string>("backend");

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }

    CombinedSolver solver(sourceMesh, targetMeshes, sourceTetIndices, params, location, backend, numthreads);
    /* CombinedSolver solver_cpu(sourceMesh, targetMeshes, sourceTetIndices, params, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    SimpleMesh* res = solver.result();
    
	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
    
    for (SimpleMesh* mesh : targetMeshes) {
        delete mesh;
    }
    delete sourceMesh;

	return 0;
}
