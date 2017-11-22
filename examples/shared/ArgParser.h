#pragma once
#include <boost/program_options.hpp>
#include <string.h>

namespace po = boost::program_options;
class ArgParser {
  public:
    ArgParser() {
      // Declare the supported options.
      /* po::options_description desc("Allowed options"); */
      m_desc = new po::options_description("Allowed Options");

      m_desc->add_options()
        ("help", "produce help message")
        ("backend", po::value<std::string>()->default_value("backend_cpu"), "set backend to 'backend_cuda', 'backend_cpu' or 'backend_cpu_mt'")
        ("numthreads", po::value<int>()->default_value(1), "set the number of threads (only has effect for backend_cpu_mt)")

        ("nIterations", po::value<int>()->default_value(1), "number of non-linear iterations.")
        ("lIterations", po::value<int>()->default_value(1), "number of linear iterations.")

        ("width", po::value<int>()->default_value(640), "width of the image (pixels), ignored for graph-examples")
        ("stride_x", po::value<int>()->default_value(1), "example, if this is 3, then only every 3rd pixel in x-direction of the input image is loaded into Opt, see also stride arg")

        ("height", po::value<int>()->default_value(360), "height of the image (pixels), ignored for graph-examples")
        ("stride_y", po::value<int>()->default_value(1), "example, if this is 3, then only every 3rd pixel in y-direction of the input image is loaded into Opt, see also stride arg")

        ("numvertices", po::value<int>()->default_value(-1), "number of vertices in the graph, ignored for image examples")
        ("stride", po::value<int>()->default_value(1), "stride in both x and y direction")
        ("numSubdivides", po::value<int>()->default_value(0), "number of subdivision in the mesh")
      ;
      


    }

    void parse(int argc, const char* argv[]) {
      po::store(po::parse_command_line(argc, argv, *m_desc), m_variablemap);
      po::notify(m_variablemap);    
      
      if (m_variablemap.count("help")) {
        std::cout << *m_desc << "\n";
        exit(1);
      }
    }

    template<typename T>
      T get(std::string key) {
        return m_variablemap[key].as<T>();
      }

    
  private:
    po::options_description* m_desc;
    po::variables_map m_variablemap;
};
