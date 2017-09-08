#include "main.h"
#include "CombinedSolver.h"

#include <boost/program_options.hpp>

static void loadConstraints(std::vector<std::vector<int> >& constraints, std::string filename) {
  std::ifstream in(filename, std::fstream::in);

	if(!in.good())
	{
		std::cout << "Could not open marker file " << filename << std::endl;
		assert(false);
	}

	unsigned int nMarkers;
	in >> nMarkers;
	constraints.resize(nMarkers);
	for(unsigned int m = 0; m<nMarkers; m++)
	{
		int temp;
		for (int i = 0; i < 4; ++i) {
			in >> temp;
			constraints[m].push_back(temp);
		}

	}

	in.close();
}


namespace po = boost::program_options;
int main(int argc, const char * argv[]) {
	// CAT 
        

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "produce help message")
      ("backend", po::value<std::string>()->default_value("backend_cpu"), "set backend to 'backend_cuda', 'backend_cpu' or 'backend_cpu_mt'")
      ("numthreads", po::value<int>()->default_value(1), "set the number of threads (only has effect for backend_cpu_mt)")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
    
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    
    /* if (vm.count("compression")) { */
    /*   cout << "Compression level was set to " */ 
    /*   << vm["compression"].as<int>() << ".\n"; */
    /* } else { */
    /*   cout << "Compression level was not set.\n"; */
    /* } */
    



  
    std::string filename = "../data/cat512.png";

    int downsampleFactor = 1;
	bool lmOnlyFullSolve = false;
    /* if (argc > 1) { */
    /*     filename = argv[1]; */
    /* } */
    /* if (argc > 2) { */
    /*     downsampleFactor = std::max(0,atoi(argv[2])); */ 
    /* } */
    bool performanceRun = false;
    /* if (argc > 3) { */
    /*     if (std::string(argv[3]) == "perf") { */
    /*         performanceRun = true; */
    /*         if (atoi(argv[2]) > 0) { */
    /*             lmOnlyFullSolve = true; */
    /*         } */
    /*     } else { */
    /*         printf("Invalid third parameter: %s\n", argv[3]); */
    /*     } */
    /* } */

    // Must have a mask and constraints file in the same directory as the input image
    std::string maskFilename = filename.substr(0, filename.size() - 4) + "_mask.png";
    std::string constraintsFilename = filename.substr(0, filename.size() - 3) + "constraints";
    std::vector<std::vector<int>> constraints;
    loadConstraints(constraints, constraintsFilename);

    ColorImageR8G8B8A8 image = LodePNG::load(filename);
    const ColorImageR8G8B8A8 imageMask = LodePNG::load(maskFilename);

    ColorImageR32G32B32 imageColor(image.getWidth() / downsampleFactor, image.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < image.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < image.getWidth() / downsampleFactor; x++) {
            auto val = image(x*downsampleFactor, y*downsampleFactor);

            imageColor(x,y) = vec3f(val.x, val.y, val.z);
        }
    }

    ColorImageR32 imageR32(imageColor.getWidth(), imageColor.getHeight());
    printf("width %d, height %d\n", imageColor.getWidth(), imageColor.getHeight());
    for (unsigned int y = 0; y < imageColor.getHeight(); y++) {
        for (unsigned int x = 0; x < imageColor.getWidth(); x++) {
            imageR32(x, y) = imageColor(x, y).x;
		}
	}
    int activePixels = 0;

    ColorImageR32 imageR32Mask(imageMask.getWidth() / downsampleFactor, imageMask.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < imageMask.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < imageMask.getWidth() / downsampleFactor; x++) {
            imageR32Mask(x, y) = imageMask(x*downsampleFactor, y*downsampleFactor).x;
            if (imageMask(x*downsampleFactor, y*downsampleFactor).x == 0.0f) {
                ++activePixels;
            }
		}
	}
    printf("numActivePixels: %d\n", activePixels);
	
    for (auto& constraint : constraints) {
        for (auto& c : constraint) {
            c /= downsampleFactor;
        }
    }

    for (unsigned int y = 0; y < imageColor.getHeight(); y++)
	{
        for (unsigned int x = 0; x < imageColor.getWidth(); x++)
		{
            if (y == 0 || x == 0 || y == (imageColor.getHeight() - 1) || x == (imageColor.getWidth() - 1))
			{
				std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
				constraints.push_back(v);
			}
		}
	}

    CombinedSolverParameters params;

    params.numIter = 19; // original
    /* params.numIter = 1; */

    params.useCUDA = false;

    /* params.nonLinearIter = 8; // original */
    params.nonLinearIter = 8; // original

    /* params.linearIter = 400; // original */
    params.linearIter = 2; 

    if (performanceRun) {
        params.useCUDA = false;
        params.useOpt = true;
        params.useOptLM = true;
        params.useCeres = true;
        params.earlyOut = true;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useOpt = false;
        params.useOptLM = true;
        params.linearIter = 500;
        if (image.getWidth() > 1024) {
            params.nonLinearIter = 100;
        }
        // TODO: Remove for < 2048x2048
#if !USE_CERES_PCG
        //m_params.useCeres = false;
#endif
    }
   


        /* int numthreads = 8; */
        int numthreads = vm["numthreads"].as<int>();

        /* std::string backend = "backend_cpu_mt"; */
        std::string backend = vm["backend"].as<std::string>();
        /* std::string backend = "backend_cuda"; */

        OptImage::Location location;
        if (backend == "backend_cuda") {
          location = OptImage::Location::GPU;
        } else {
          location = OptImage::Location::CPU;
        }

	/* CombinedSolver solver(imageR32, imageColor, imageR32Mask, constraints, params, OptImage::Location::GPU, "backend_cuda", numthreads); */
	/* CombinedSolver solver_cpu(imageR32, imageColor, imageR32Mask, constraints, params, OptImage::Location::CPU, "backend_cpu_mt", numthreads); */

	CombinedSolver solver(imageR32, imageColor, imageR32Mask, constraints, params, location, backend, numthreads);

    solver.solveAll();
    /* solver_cpu.solveAll(); */

    ColorImageR32G32B32* res = solver.result();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = util::boundToByte((*res)(x, y).x);
            unsigned char g = util::boundToByte((*res)(x, y).y);
            unsigned char b = util::boundToByte((*res)(x, y).z);
			out(x, y) = vec4uc(r, g, b, 255);
	
			for (unsigned int k = 0; k < constraints.size(); k++)
			{
				if (constraints[k][2] == x && constraints[k][3] == y) 
				{
                    if (imageR32Mask(constraints[k][0], constraints[k][1]) == 0)
					{
						//out(x, y) = vec4uc(255, 0, 0, 255);
					}
				}
		
				if (constraints[k][0] == x && constraints[k][1] == y)
				{
					if (imageR32Mask(x, y) == 0)
					{
                        image(x*downsampleFactor, y*downsampleFactor) = vec4uc(255, 0, 0, 255);
					}
				}
			}
		}
	}
	LodePNG::save(out, "output.png");
	LodePNG::save(image, "inputMark.png");
    printf("Saved\n");

	return 0;
}
