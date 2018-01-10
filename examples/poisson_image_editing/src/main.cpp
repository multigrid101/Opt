#include "main.h"
#include "CombinedSolver.h"
#include "../../shared/ArgParser.h"

int main(int argc, const char * argv[]) {
	std::string inputImage0 = "../data/poisson0.png";
	std::string inputImage1 = "../data/poisson1.png";
	std::string inputImageMask = "../data/poisson_mask.png";
    /* if (argc > 1) { */
    /*     assert(argc > 3); */
    /*     inputImage0 = argv[1]; */
    /*     inputImage1 = argv[2]; */
    /*     inputImageMask = argv[3]; */
    /* } */
        ArgParser argparser;
        argparser.parse(argc, argv);


	const unsigned int offsetX = 0;
	const unsigned int offsetY = 0;
	const bool invertMask = false;


    int stride = argparser.get<int>("stride");
    printf("stride is %d\n", stride);




    ColorImageR8G8B8A8	   image = LodePNG::load(inputImage0);
    int targetWidth = image.getWidth()/stride;
    int targetHeight = image.getHeight()/stride;


	/* ColorImageR32G32B32A32 imageR32(image.getWidth(), image.getHeight()); */
	ColorImageR32G32B32A32 imageR32(targetWidth, targetHeight);
	for (unsigned int y = 0; y < targetHeight; y++) {
		for (unsigned int x = 0; x < targetWidth; x++) {
			imageR32(x,y) = image(stride*x,stride*y);
		}
	}

	ColorImageR8G8B8A8	   image1 = LodePNG::load(inputImage1);
	ColorImageR32G32B32A32 imageR321(targetWidth, targetHeight);
	for (unsigned int y = 0; y < targetHeight; y++) {
		for (unsigned int x = 0; x < targetWidth; x++) {
			imageR321(x, y) = image1(stride*x, stride*y);
		}
	}

	ColorImageR32G32B32A32 image1Large = imageR32;
	image1Large.setPixels(ml::vec4uc(0, 0, 0, 255));
	for (unsigned int y = 0; y < imageR321.getHeight(); y++) {
		for (unsigned int x = 0; x < imageR321.getWidth(); x++) {
			image1Large(x + offsetY, y + offsetX) = imageR321(x, y);
		}
	}


	
	const ColorImageR8G8B8A8 imageMask = LodePNG::load(inputImageMask);
	ColorImageR32 imageR32Mask(targetWidth, targetHeight);
	for (unsigned int y = 0; y < targetHeight; y++) {
		for (unsigned int x = 0; x < targetWidth; x++) {
			unsigned char c = imageMask(stride*x, stride*y).x;
			if (invertMask) {
				if (c == 255) c = 0;
				else c = 255;
			}

			imageR32Mask(x, y) = c;
		}
	}

	ColorImageR32 imageR32MaskLarge(targetWidth, targetHeight);
	imageR32MaskLarge.setPixels(0);
	for (unsigned int y = 0; y < targetHeight; y++) {
		for (unsigned int x = 0; x < targetWidth; x++) {
			imageR32MaskLarge(x + offsetY, y + offsetX) = imageR32Mask(stride*x, stride*y);
		}
	}
	
    CombinedSolverParameters params;

    params.useOpt = true;

    /* params.nonLinearIter = 1; // original */
    params.nonLinearIter = argparser.get<int>("nIterations");

    /* params.linearIter = 100; //original */
    params.linearIter = argparser.get<int>("lIterations");

    int numthreads = argparser.get<int>("numthreads");
    std::string backend = argparser.get<std::string>("backend");

    params.useOpt = argparser.get<bool>("useOpt");
    params.useOptLM = argparser.get<bool>("useOptLM");
    params.useCeres = argparser.get<bool>("useCeres");

    params.useMaterializedJTJ = argparser.get<bool>("useMaterializedJTJ");
    params.useFusedJTJ = argparser.get<bool>("useFusedJTJ");

    OptImage::Location location;
    if (backend == "backend_cuda") {
      location = OptImage::Location::GPU;
    } else {
      location = OptImage::Location::CPU;
    }

    // This example has a couple solvers that don't fit into the CombinedSolverParameters mold.
    bool useCUDAPatch = false;
    bool useEigen = false;

    CombinedSolver solver(imageR32, image1Large, imageR32MaskLarge, params, useCUDAPatch, useEigen, location, backend, numthreads);
    /* CombinedSolver solver_cpu(imageR32, image1Large, imageR32MaskLarge, params, useCUDAPatch, useEigen, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    ColorImageR32G32B32A32* res = solver.result();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(math::clamp((*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(math::clamp((*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(math::clamp((*res)(x, y).z, 0.0f, 255.0f));
			out(x, y) = vec4uc(r, g, b,255);
		}
	}
	LodePNG::save(out, "output.png");
	return 0;
}
