#ifndef SFSSolverInput_h
#define SFSSolverInput_h
#include "SimpleBuffer.h"
#include "TerraSolverParameters.h"
#include "../../shared/NamedParameters.h"
#include <memory>
#include <string>
static  std::shared_ptr<OptImage> createWrapperOptImage(std::shared_ptr<SimpleBuffer> simpleBuffer, OptImage::Location location) {

    // get width and height from simpleBuffer
    std::vector<unsigned int> dims = { (unsigned int)simpleBuffer->width(), (unsigned int)simpleBuffer->height() };

    // get datatype
    OptImage::Type t = (simpleBuffer->type() == SimpleBuffer::DataType::FLOAT) ? OptImage::Type::FLOAT : OptImage::Type::UCHAR;
    bool isUnknown = (t == OptImage::Type::FLOAT);

    // wrap simpleBuffer data in OptImage
    return std::shared_ptr<OptImage>(new OptImage(dims, simpleBuffer->data(), t, 1, location, isUnknown, false));
}

struct SFSSolverInput {
    std::shared_ptr<SimpleBuffer>   targetIntensity;
    std::shared_ptr<SimpleBuffer>   targetDepth;
    std::shared_ptr<SimpleBuffer>   initialUnknown; // The values to initialize d_x to before the solver
    std::shared_ptr<SimpleBuffer>   maskEdgeMap; //uint8s, and actually the row and column maps stuck together...
    TerraSolverParameters           parameters;

    void setParameters(NamedParameters& probParams, std::shared_ptr<SimpleBuffer> unknownImage, OptImage::Location location) const {
        probParams.set("w_p", (void*)&parameters.weightFitting);
        probParams.set("w_s", (void*)&parameters.weightRegularizer);
        probParams.set("w_g", (void*)&parameters.weightShading);
        probParams.set("f_x", (void*)&parameters.fx);
        probParams.set("f_y", (void*)&parameters.fy);
        probParams.set("u_x", (void*)&parameters.ux);
        probParams.set("u_y", (void*)&parameters.uy);
        for (int i = 0; i < 9; ++i) {
            char buff[5];
            sprintf(buff, "L_%d", i+1);
            probParams.set(buff, (void*)&(parameters.lightingCoefficients[i]));
        }
        
        auto unknown = createWrapperOptImage(unknownImage, location);
        probParams.set("X", unknown);

        auto di = createWrapperOptImage(targetDepth, location);
        probParams.set("D_i", di);

        probParams.set("Im", createWrapperOptImage(targetIntensity, location));

        std::shared_ptr<OptImage> edgeMaskR = createEmptyOptImage(unknown->dims(), OptImage::Type::UCHAR, 1, location, false);
        std::shared_ptr<OptImage> edgeMaskC = createEmptyOptImage(unknown->dims(), OptImage::Type::UCHAR, 1, location, false);

        size_t pixCount = initialUnknown->width()*initialUnknown->height();

        /* edgeMaskR->update(maskEdgeMap->data(), pixCount*sizeof(unsigned char), OptImage::Location::GPU); */
        /* edgeMaskC->update((unsigned char*)maskEdgeMap->data() + pixCount, pixCount*sizeof(unsigned char), OptImage::Location::GPU); */
        edgeMaskR->update(maskEdgeMap->data(), pixCount*sizeof(unsigned char), location);
        edgeMaskC->update((unsigned char*)maskEdgeMap->data() + pixCount, pixCount*sizeof(unsigned char), location);

        probParams.set("edgeMaskR", edgeMaskR);
        probParams.set("edgeMaskC", edgeMaskC);
    }

    void load(const std::string& filenamePrefix, bool onGPU) {
        targetIntensity = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetIntensity.imagedump", onGPU));
        targetDepth     = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetDepth.imagedump",     onGPU));
        initialUnknown  = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_initialUnknown.imagedump", onGPU));
        maskEdgeMap     = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_maskEdgeMap.imagedump",     onGPU));
	
        auto test = std::shared_ptr<SimpleBuffer>(new SimpleBuffer(filenamePrefix + "_targetDepth.imagedump", false));
        float* ptr = (float*)test->data();
        int numActiveUnkowns = 0;
        for (int i = 0; i < test->width()*test->height(); ++i) {
            if (ptr[i] > 0.0f) {
                ++numActiveUnkowns;
            }
        }
        printf("Num Active Unknowns: %d\n", numActiveUnkowns);

        parameters.load(filenamePrefix + ".SFSSolverParameters");
    }

};

#endif
