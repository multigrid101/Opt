#include "main.h"
#include "CombinedSolver.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/ArgParser.h"


int main(int argc, const char * argv[])
{
    std::string filename = "../data/ye_high2.png"; //original
    /* if (argc > 1) { */
    /*     filename = argv[1]; */
    /* } */
    ArgParser argparser;
    argparser.parse(argc, argv);

    int stride = argparser.get<int>("stride");
    printf("stride is %d\n", stride);

    ColorImageR8G8B8A8	   image = LodePNG::load(filename);
    int targetWidth = image.getWidth()/stride;
    int targetHeight = image.getHeight()/stride;


	ColorImageR32G32B32A32 imageR32(targetWidth, targetHeight);

	for (unsigned int y = 0; y < targetHeight; y++) {
		for (unsigned int x = 0; x < targetWidth; x++) {
			imageR32(x,y) = image(stride*x,stride*y);
		}
	}
	

    CombinedSolverParameters params;

    /* params.nonLinearIter = 7;// original */
    params.nonLinearIter = argparser.get<int>("nIterations");

    /* params.linearIter = 10; original */
    params.linearIter = argparser.get<int>("lIterations");

    int numthreads = argparser.get<int>("numthreads");
    std::string backend = argparser.get<std::string>("backend");

    params.useOpt = argparser.get<bool>("useOpt");
    params.useOptLM = argparser.get<bool>("useOptLM");
    params.useCeres = argparser.get<bool>("useCeres"); // makes no sense here

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }


    CombinedSolver solver(imageR32, params, location, backend, numthreads);
    /* CombinedSolver solver_cpu(imageR32, params, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    ColorImageR32G32B32A32* res = solver.getAlbedo();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(math::clamp(255.0f*(*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(math::clamp(255.0f*(*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(math::clamp(255.0f*(*res)(x, y).z, 0.0f, 255.0f));
			out(x, y) = vec4uc(r, g, b,255);
		}
	}
	LodePNG::save(out, "outputAlbedo.png");

    res = solver.getShading();
	ColorImageR8G8B8A8 out2(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(255.0f*math::clamp((*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(255.0f*math::clamp((*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(255.0f*math::clamp((*res)(x, y).z, 0.0f, 255.0f));
			out2(x, y) = vec4uc(r, g, b, 255);
		}
	}
	LodePNG::save(out2, "outputShading.png");
	return 0;
}
