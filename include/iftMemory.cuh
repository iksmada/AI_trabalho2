#ifdef IFT_GPU
#ifndef IFT_MEMORY_CUH
#define IFT_MEMORY_CUH

#include <cuda.h>
#include <cublas.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Init the GPU devices.
 * @warning If not called, the GPU devices will be started in the first call for a GPU function.
 * @author Peixinho
 */


void iftStartGPU();

/**
 * @brief Stop the GPU devices.
 * @author Peixinho
 */
void iftStopGPU();

void *iftAllocGPU(int n, size_t size);

void iftFreeGPU(void *mem);

void iftCopyToGPU(void* dst, void* src, size_t size);

void iftCopyFromGPU(void* dst, void* src, size_t size);

float* iftAllocFloatArrayGPU(int n);

void iftCopyFloatArrayFromGPU(float* dst, float* src, int n);

void iftCopyFloatArrayToGPU(float* dst, float* src, int n);

int* iftAllocIntArrayGPU(int n);

void iftCopyIntArrayFromGPU(int* dst, int* src, int n);

void iftCopyIntArrayToGPU(int* dst, int* src, int n);

float iftGetFreeMemoryGPU(int device, char unit);

#ifdef __cplusplus
}
#endif

#endif//IFT_MEMORY_CUH
#endif//IFT_GPU
