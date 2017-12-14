#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "ImageHelper.h"
#include "../../shared/ArgParser.h"


void renderFlowVecotors(ColorImageR8G8B8A8& image, const BaseImage<float2>& flowVectors) {
	const unsigned int skip = 5;	//only every n-th pixel
	const float lengthRed = 5.0f;
	
	for (unsigned int j = 1; j < image.getHeight() - 1; j += skip) {
		for (unsigned int i = 1; i < image.getWidth() - 1; i += skip) {
			
			const float2& flowVector = flowVectors(i, j);
			vec2i start = vec2i(i, j);
			vec2i end = start + vec2i(math::round(flowVector.x), math::round(flowVector.y));
			float len = vec2f(flowVector.x, flowVector.y).length();
			vec4uc color = math::round(255.0f*BaseImageHelper::convertDepthToRGBA(len, 0.0f, 5.0f)*2.0f);	color.w = 255;
			//vec4uc color = math::round(255.0f*vec4f(0.1f, 0.8f, 0.1f, 1.0f));	//TODO color-code length

			ImageHelper::drawLine(image, start, end, color);
		}
	}
}

int main(int argc, const char * argv[]) {

    std::string srcFile = "../data/dogdance0.png";
    std::string tarFile = "../data/dogdance1.png";

    ArgParser argparser;
    argparser.parse(argc, argv);

    /* if (argc > 1) { */
    /*     assert(argc > 2); */
    /*     srcFile = argv[1]; */
    /*     tarFile = argv[2]; */
    /* } */

	ColorImageR8G8B8A8 imageSrc_ = LodePNG::load(srcFile);
	ColorImageR8G8B8A8 imageTar = LodePNG::load(tarFile);

        // original pictures, not downscaled
	ColorImageR32 imageSrcGray_ = imageSrc_.convertToGrayscale();
	ColorImageR32 imageTarGray_ = imageTar.convertToGrayscale();


    // downscale according to the stride parameter
    int stride = argparser.get<int>("stride");

    ColorImageR8G8B8A8 imageSrc(imageSrc_.getWidth()/stride, imageSrc_.getHeight()/stride);
    ColorImageR32 imageSrcGray(imageSrcGray_.getWidth()/stride, imageSrcGray_.getHeight()/stride);
    ColorImageR32 imageTarGray(imageTarGray_.getWidth()/stride, imageTarGray_.getHeight()/stride);

    for (unsigned int j = 0; j < imageSrcGray.getHeight(); j++) {
        for (unsigned int i = 0; i < imageSrcGray.getWidth(); i++) {
            imageSrcGray(i,j) = imageSrcGray_(i*stride, j*stride);
        }
    }
    for (unsigned int j = 0; j < imageTarGray.getHeight(); j++) {
        for (unsigned int i = 0; i < imageTarGray.getWidth(); i++) {
            imageTarGray(i,j) = imageTarGray_(i*stride, j*stride);
        }
    }
    for (unsigned int j = 0; j < imageSrc.getHeight(); j++) {
        for (unsigned int i = 0; i < imageSrc.getWidth(); i++) {
            imageSrc(i,j) = imageSrc_(i*stride, j*stride);
        }
    }
    

    // define solver parameters
    CombinedSolverParameters params;

    /* params.numIter = 3; //original */
    params.numIter = argparser.get<int>("oIterations");


    /* params.nonLinearIter = 1; //original */
    params.nonLinearIter = argparser.get<int>("nIterations");


    /* params.linearIter = 50; //original */
    params.linearIter = argparser.get<int>("lIterations");

        int numthreads = argparser.get<int>("numthreads");

        std::string backend = argparser.get<std::string>("backend");

    params.useOpt = argparser.get<bool>("useOpt");
    params.useOptLM = argparser.get<bool>("useOptLM");
    params.useCeres = argparser.get<bool>("useCeres"); // makes no sense here

    params.useMaterializedJTJ = argparser.get<bool>("useMaterializedJTJ");
    params.useFusedJTJ = argparser.get<bool>("useMaterializedJTJ");

        OptImage::Location location;
        if (backend == "backend_cuda") {
          location = OptImage::Location::GPU;
        } else {
          location = OptImage::Location::CPU;
        }

    CombinedSolver solver(imageSrcGray, imageTarGray, params, location, backend, numthreads);
    /* CombinedSolver solver_cpu(imageSrcGray, imageTarGray, params, OptImage::Location::CPU); */

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    BaseImage<float2> flowVectors = solver.result();

	const std::string outFile = "out.png";
	ColorImageR8G8B8A8 out = imageSrc;
	renderFlowVecotors(out, flowVectors);
	LodePNG::save(out, outFile);

	return 0;
}
