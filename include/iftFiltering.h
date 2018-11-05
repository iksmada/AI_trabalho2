#ifndef IFT_FILTERING_H_
#define IFT_FILTERING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "iftCommon.h"
#include "iftImage.h"
#include "iftFImage.h"
#include "iftMImage.h"
#include "iftAdjacency.h"
#include "iftSort.h"
#include "iftMatrix.h"
#include "iftKernel.h"

//! swig(newobject)
iftImage  *iftLinearFilter(const iftImage *img, iftKernel *K);
//! swig(newobject)
iftImage  *iftLinearFilterInRegion(iftImage *img, iftImage *mask, iftKernel *K);
//! swig(newobject)
iftImage  *iftFastLinearFilter(iftImage *img, iftKernel *K, char crop);
iftImage  *iftCroppedFastLinearFilterByMatrixMult(iftImage *img, iftKernel *K);
//! swig(newobject)
iftFImage *iftFLinearFilter(iftFImage *img, iftKernel *K);
//! swig(newobject)
iftFImage  *iftFLinearFilterInRegion(iftFImage *img, iftImage *mask, iftKernel *K);
//! swig(newobject)
iftFImage *iftFastFLinearFilter(iftFImage *img, iftKernel *K, char crop);
//! swig(newobject)
iftMImage *iftMLinearFilter(iftMImage *img, iftMKernel *K);
//! swig(newobject)
iftMImage *iftMMLinearFilter(iftMImage *img, iftMMKernel *k_bank);

/**
 * @brief Apply the Median Filter on an Input Image.
 * @param  img Input Image to be filtered.
 * @param  Ain Input adjacent used in filtering. If NULL, a 4-neighborhood (2D) or
 *             6-neighborhood (3D) is considered.
 * @return     Filtered Image.
 */
//! swig(newobject)
iftImage *iftMedianFilter(const iftImage *img, iftAdjRel *Ain);


/**
 * @brief Applies a Median filter on an Input Image by considering all neighbors from an adjacency
 * relation in the computation.
 * 
 * @param img Image to be filtered.
 * @param Ain Adjacency Relation. If NULL, a 4-neighborhood (2D) or
 *            neighborhood (3D) is considered.
 * 
 * @return Filtered image.
 * 
 * @author Samuel Martins
 * @date Sep 14, 2018
 */
iftImage *iftMeanFilter(const iftImage *img, const iftAdjRel *Ain);


//! swig(newobject)
iftMImage *iftMMedianFilter(iftMImage *img, iftAdjRel *A);
//! swig(newobject)
iftImage  *iftModaFilter(iftImage *img, iftAdjRel *A);
//! swig(newobject)
iftImage  *iftSobelGradientMagnitude(iftImage *img);
//! swig(newobject)
iftMatrix *iftImageToMatrix(iftImage *img, iftFastAdjRel *F, char crop);
//! swig(newobject)
iftMatrix *iftImageAdjToMatrix(iftImage *img, iftAdjRel *A);
//! swig(newobject)
iftImage  *iftMatrixToImage(iftMatrix *M, int xsize, int ysize, int zsize);
//! swig(newobject)
iftMatrix *iftFImageToMatrix(iftFImage *img, iftFastAdjRel *F, char crop);
//! swig(newobject)
iftFImage  *iftMatrixToFImage(iftMatrix *M, int xsize, int ysize, int zsize);
//! swig(newobject)
iftMatrix *iftMImageToMatrix(iftMImage *img, iftFastAdjRel *F);
//! swig(newobject)
iftMImage *iftMatrixToMImage(iftMatrix *M, int xsize, int ysize, int zsize, int nbands);
//! swig(newobject)
iftMatrix *iftKernelToMatrix(iftKernel *K);
//! swig(newobject)
iftMatrix *iftMKernelToMatrix(iftMKernel *K);
//! swig(newobject)
iftMatrix *iftMMKernelToMatrix(iftMMKernel *k_bank);
//! swig(newobject)
iftImage  *iftSmoothImage(iftImage *img, iftAdjRel *A, float sigma);
//! swig(newobject)
iftImage  *iftSmoothImageInRegion(iftImage *img, iftImage *mask, iftAdjRel *A, float sigma);

void iftFastBilateralFilter2DAux(const float * __restrict__ _input,
                                 float       * __restrict__ _output,
                                 int   width,
                                 int   height,
                                 int 	channels,
                                 int   s_sigma,
                                 float r_sigma);

//! swig(newobject)
iftImage *iftFastBilateralFilter2D(iftImage *img, int s_sigma, float r_sigma);
//! swig(newobject)
iftMImage *iftFastMBilateralFilter2D(iftMImage *img, int s_sigma, float r_sigma);

//! swig(newobject)
iftImage *iftNormalizeImage(iftImage *img, iftAdjRel *A, int Imax);
//! swig(newobject)
iftImage *iftAlphaPooling(iftImage *img, iftAdjRel *A, int stride, float alpha);



#ifdef __cplusplus
}
#endif

#endif
