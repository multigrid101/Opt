# CXX = clang++
CXX = g++
# LINK = clang++
LINK = g++
NVCC = /usr/local/cuda/bin/nvcc
SRC ?= src
CUDAHOME += /usr/local/cuda
TERRAHOME ?= ../../../terra
TERRA ?= ../../../terra/release/terra
SHARED ?= ../shared

ifeq ($(CXX), clang++)
  OMPLFLAG = -fopenmp=libiomp5
else
  OMPLFLAG = -fopenmp
endif

ifeq ($(UNAME), Linux)
  S4 = 64
endif

FLAGS += -O3 -g -std=c++11 -Isrc -Isrc/cutil/inc -I../../API/release/include -I$(TERRAHOME)/include -I$(CUDAHOME)/include -I../external/mLib/include -I../external -I../external/OpenMesh/include -I/home/sebastian/Desktop/ceres-solver-1.13.0/include/ -I/usr/include/eigen3 -I/home/sebastian/Desktop/ceres-solver-1.13.0/config/
NVCCFLAGS += -arch sm_30 $(FLAGS)
LFLAGS += -std=c++11 -L/home/sebastian/Desktop/ceres-bin/lib -lceres -L../../API/release/lib -L$(CUDAHOME)/lib$(S4) -lcudart -L$(TERRAHOME)/lib -lterra -ldl -lboost_program_options -L /opt/intel/vtune_amplifier_xe_2017/lib64 -littnotify -lglog -lgflags $(OMPLFLAG) -lgomp -lm -llapack -lcamd -lamd -lccolamd -lcolamd -lcholmod -lblas -lcxsparse -littnotify
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  LFLAGS += -pagezero_size 10000 -image_base 100000000
endif

ifeq ($(UNAME), Linux)
  LFLAGS += -Wl,-rpath,$(CUDAHOME)/lib$(S4) -pthread
  S4 = 64
endif

.SUFFIXES:
.PHONY:	all purge clean

all: $(EXECUTABLE)

OPTLIB = ../../API/release/lib/libOptDev.a
CERESLIB = /home/sebastian/Desktop/ceres-bin/lib/libceres.a

build/%.o:	$(SRC)/%.cpp
	@mkdir -p build
	$(CXX) $(FLAGS) -fpermissive -MP -MD $< -c -o $@
build/shared/%.o:	$(SHARED)/%.cpp
	@mkdir -p build
	@mkdir -p build/shared
	$(CXX) $(FLAGS) -fpermissive -MP -MD $< -c -o $@
build/%.o:	$(SRC)/%.cu
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(EXECUTABLE):	$(OBJS) $(OPTLIB) $(CERESLIB)
	$(CXX) $^ $(LFLAGS) -o $@ -rdynamic

clean:
	rm -rf build/*.o build/*.d
	rm -rf $(EXECUTABLE)

purge: clean
	rm -rf build/*

# dependency rules
include $(wildcard build/*.d)
