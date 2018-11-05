#ifdef IFT_GPU
#ifndef IFT_MATRIX_CUH
#define IFT_MATRIX_CUH

#include <iftMatrix.h>
#include <iftMemory.cuh>
#include <cuda.h>
#include <cublas.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiplies matrices A*B in GPU device.
 * @author Peixinho
 * @date Jan 2017
 * @warning This function is intended to be used two multiply large matrices (Prefereably with all the dimensions above 600). If iftInitGPU() is not called, the initialization of GPU will be called here, what may cause some overhead.
 */
void iftMultMatricesInPlaceGPU(const iftMatrix *A, const iftMatrix *B, bool transposeA, bool transposeB, iftMatrix **C);

void iftCopyTensorToGPU(iftTensor* dst, iftTensor* src, int n);

iftTensor* iftAllocTensorGPU(int n);

#ifdef __cplusplus
}
#endif

#endif//IFT_MATRIX_CUH
#endif//IFT_GPU
