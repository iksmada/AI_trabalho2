/**
 * @brief Memory Management functions.
 * @author Peixinho
 * @date Oct 8, 2015
 * @ingroup Memory
 */

#ifndef IFT_IFTMEMORY_H
#define IFT_IFTMEMORY_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__APPLE__)
    #include <malloc.h>
#endif


#include "iftBasicDataTypes.h"

/**
 * @brief Array of uchar values.
 * @author Thiago Vallin Spina
 * @date Mar 13, 2016
 * @ingroup Memory
 */
typedef struct ift_uchar_array {
    /** Number of elements. */
    long n;
    /** Array of uchar values. */
    uchar *val;
} iftUCharArray;


/**
 * @brief Array of integer values.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 * 
 * @note Its type definition in iftBasicDataType.h
 */
//! swig(extend = iftIntArrayExt.i, destroyer = iftDestroyIntArray, name = iftIntArray)
struct ift_int_array {
    /** Number of elements. */
    long n;
    /** Array of integer values. */
    int *val;
};


/**
 * @brief Array of unsigned long values.
 * @author Thiago Vallin spina
 * @date Mar 1, 2016
 * @ingroup Memory
 */
typedef struct ift_ulong_array {
    /** Number of elements. */
    long n;
    /** Array of integer values. */
    ulong *val;
} iftULongArray;


/**
 * @brief Array of long values.
 * @author Thiago Vallin spina
 * @date Mar 1, 2016
 * @ingroup Memory
 */
typedef struct ift_long_array {
    /** Number of elements. */
    long n;
    /** Array of integer values. */
    long *val;
} iftLongArray;


/**
 * @brief Array of float values.
 * @author Thiago Vallin Spina
 * @date Mar 4, 2016
 * @ingroup Memory
 */
//! swig(destroyer = iftDestroyFloatArray)
typedef struct ift_flt_array {
    /** Number of elements. */
    long n;
    /** Array of float values. */
    float *val;
} iftFloatArray;


/**
 * @brief Array of double values.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
struct ift_dbl_array {
    /** Number of elements. */
    long n;
    /** Array of double values. */
    double *val;
};


/**
 * @brief Array of strings.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 * 
 * @note Its type definition in iftBasicDataType.h
 */
struct ift_str_array {
    /** Number of elements. */
    long n;
    /** Array of strings. */
    char **val;
};


/**
 * @brief Array of Voxels.
 * @author Samuka Martins
 * @date Aug 23, 2018
 * @ingroup Memory
 */
typedef struct ift_voxel_arr {
    /** Number of Elements */
    long n;
    /** Array of Voxels */
    iftVoxel *val;
} iftVoxelArray;


/**
 * @brief Invoke garbage collector.
 * @author Peixinho
 * @ingroup Memory
 * @date Nov, 2016
 */
void iftGC();

/**
 * @brief The amount of memory currently being used by this process, in bytes.
 * @author Peixinho
 * @ingroup Memory
 */
long iftMemoryUsed();
long iftGetPhysicalSystemMemory();

/**
 * @brief Shows the current number of allocated objects, for debugging purpose. See also iftAllocObjectsCount().
 * @author Peixinho
 * @ingroup Memory
 */
void iftPrintAllocObjectsCount();

/**
 * @brief Prints a memory size, with an appropriate metric (KB, MB, GB).
 * @param mem Memory size in bytes
 * @author Peixinho
 * @date Jun, 2017
 */
void iftPrintMemoryFormatted(long mem);

/**
 * @brief Count the current number of allocated objects, for debugging purpose.
 * @author Peixinho
 * @ingroup Memory
 */
long iftAllocObjectsCount();

/**
 * @brief Prints the memory amount used by a array.
 * @author Cesar Castelo
 * @param arraySize Size of the array
 * @param elemSize Size of each element
 * @param unit A char indicating the unit of measure (b: bytes, k: kbytes, m: mbytes, g: gbytes)
 * @ingroup Memory
 */
double iftMemoryUsedByArray(long arraySize, int elemSize, char unit);

/**
 * @brief Allocate n memory blocks of size sz. See also iftFree().
 * @ingroup Memory
 * @author Peixinho
 */
void* iftAlloc(long n, long sz);

/**
 * @brief Releases a memory block previously allocated. See also iftAlloc().
 * @ingroup Memory
 * @author Peixinho
 */
void iftFree(void* mem);

/**
 * @brief Allocate reusing memory n blocks of size sz. See also iftFree().
 * @ingroup Memory
 * @author Peixinho
 */
void* iftRealloc(void* p, long size);

/**
 * @brief  Verify memory deallocation. 
 * @author Alexandre Falcao
 * @date   Feb 23, 2016
 * @ingroup Memory
 */
bool iftVerifyMemory(long MemDinInicial, long MemDinFinal);

/**
 * @brief Swap the content of two variables.
 * @warning In pointer, this function only swaps the pointers themselves, not their memory content.
 * @author Peixinho
 * @date Mar, 2016
 */
#define iftSwap(x, y) do { __typeof__(x) _IFT_SWAP_ = x; x = y; y = _IFT_SWAP_; } while (0)

/**
 * @brief Swap the content of two strings.
 * @author Peixinho
 * @date Mar, 2016
 */
void iftSwapString(char *a, char *b);


/**
 * @brief Shuffles an integer array by the modern version Fisher-Yates shuffle algorithm.
 * @details See https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
 * @param array Integer array
 * @param n Size of the array
 */
void iftShuffleIntArray(int* array, int n);

/**
 * @brief Allocates a Char Array (string) with <b>n</b> positions
 * @ingroup Memory
 * @{
 */
bool *iftAllocBoolArray(long n);
char *iftAllocCharArray(long n);
char *iftAllocString(long n);
uchar *iftAllocUCharArray(long n);
short *iftAllocShortArray(long n);
ushort *iftAllocUShortArray(long n);
int *iftAllocIntArray(long n);
uint *iftAllocUIntArray(long n);
long *iftAllocLongIntArray(long n);
#ifndef  __cplusplus
long long *iftAllocLongLongIntArray(long n);
#endif
ulong *iftAllocULongArray(long n);
#ifndef  __cplusplus
ullong *iftAllocULLongArray(long n);
#endif
float *iftAllocFloatArray(long n);
double *iftAllocDoubleArray(long n);
long double *iftAllocLongDoubleArray(long n);
iftComplex *iftAllocComplexArray(long n);

// Aligned memory allocation
uchar *iftAllocAlignedUCharArray(long n, long alignment);
int *iftAllocAlignedIntArray(long n, long alignment);
float *iftAllocAlignedFloatArray(long n, long alignment);
double *iftAllocAlignedDoubleArray(long n, long alignment);
/** @} */


/**
 * @brief Functions to allocate/deallocate a matrix with <b>n</b> positions of different data types.
 * The matrix is allocated as a single array of size (c x r) with the corresponding pointers to the beginning of each row
 * @ingroup Memory
 * @author Cesar Castelo
 * @date Jul 16, 2018
 * @warning These pointers MUST be set free with iftFreeMatrix. A call to iftFree DOES NOT free the memory properly
 * @{
 */
bool **iftAllocBoolMatrix(long c, long r);
char **iftAllocCharMatrix(long c, long r);
uchar **iftAllocUCharMatrix(long c, long r);
short **iftAllocShortMatrix(long c, long r);
ushort **iftAllocUShortMatrix(long c, long r);
int **iftAllocIntMatrix(long c, long r);
uint **iftAllocUIntMatrix(long c, long r);
long **iftAllocLongMatrix(long c, long r);
#ifndef  __cplusplus
long long **iftAllocLongLongMatrix(long c, long r);
#endif
ulong **iftAllocULongMatrix(long c, long r);
#ifndef  __cplusplus
ullong **iftAllocULLongMatrix(long c, long r);
#endif
float **iftAllocFloatMatrix(long c, long r);
double **iftAllocDoubleMatrix(long c, long r);
long double **iftAllocLongDoubleMatrix(long c, long r);
iftComplex **iftAllocComplexMatrix(long c, long r);

#define iftFreeMatrix(M, NROWS) {\
    if(M!=NULL) {\
        for(int i = 0; i < NROWS; i++) {\
            if(M[i]!=NULL) {\
                iftFree(M[i]);\
            }\
        }\
        iftFree(M);\
    }\
}
/** @} */


/**
 * @brief Creates an iftUCharArray.
 * @author Thiago Vallin Spina
 * @date Mar 13, 2016
 * @ingroup Memory
 */
iftUCharArray *iftCreateUCharArray(long n);


/**
 * @brief Destroys an iftUCharArray.
 * @author Thiago Vallin Spina
 * @date Mar 13, 2016
 * @ingroup Memory
 */
void iftDestroyUCharArray(iftUCharArray **iarr);


/**
 * @brief Reallocates memory for an iftUCharArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeUCharArray(iftUCharArray **iarr, long n);

/**
 * @brief Creates an iftIntArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
iftIntArray *iftCreateIntArray(long n);


/**
 * @brief Destroys an iftIntArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
void iftDestroyIntArray(iftIntArray **iarr);


/**
 * @brief Reallocates memory for an iftIntArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeIntArray(iftIntArray **iarr, long n);


/**
 * @brief Converts a voxel v to an Integer Array of size 3.
 * [0] = x, [1] = y, [2] = z
 *
 * @author Samuka
 * @date Nov 16, 2017
 */
iftIntArray *iftVoxelToIntArray(iftVoxel v);


/**
 * @brief Converts an integer array of size 3 to a voxel
 * [0] = x, [1] = y, [2] = z
 *
 * @author Samuka
 * @date Nov 16, 2017
 */
iftVoxel iftIntArrayToVoxel(const iftIntArray *arr);


/**
 * @brief Rearrange an array to a string for printing.
 * @author Samuka;
 * @date Sep 16, 2017
 */
char *iftIntArrayAsString(const iftIntArray *iarr);


/**
 * @brief Creates an iftULongArray.
 * @author Thiago Vallin Spina
 * @date Feb 15, 2016
 * @ingroup Memory
 */
iftULongArray *iftCreateULongArray(long n);


/**
 * @brief Destroys an iftULongArray.
 * @author Thiago Vallin Spina
 * @date Feb 15, 2016
 * @ingroup Memory
 */
void iftDestroyULongArray(iftULongArray **iarr);


/**
 * @brief Reallocates memory for an iftULongArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeULongArray(iftULongArray **iarr, long n);


/**
 * @brief Creates an iftLongArray.
 * @author Thiago Vallin Spina
 * @date Feb 15, 2016
 * @ingroup Memory
 */
iftLongArray *iftCreateLongArray(long n);


/**
 * @brief Destroys an iftLongArray.
 * @author Thiago Vallin Spina
 * @date Feb 15, 2016
 * @ingroup Memory
 */
void iftDestroyLongArray(iftLongArray **iarr);


/**
 * @brief Reallocates memory for an iftLongArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeLongArray(iftLongArray **iarr, long n);


/**
 * @brief Creates an iftFloatArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
iftFloatArray *iftCreateFloatArray(long n);


/**
 * @brief Destroys an iftDblArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
void iftDestroyFloatArray(iftFloatArray **darr);


/**
 * @brief Reallocates memory for an iftFloatArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeFloatArray(iftFloatArray **iarr, long n);


/**
 * @brief Rearrange an array to a string for printing.
 * @author Samuka;
 * @date Dec 28, 2017
 */
char *iftFloatArrayAsString(const iftFloatArray *farr);


/**
 * @brief Creates an iftDblArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
iftDblArray *iftCreateDblArray(long n);

/**
 * Copies a Double array, from data.
 * @author Peixinho
 * @date April, 2016
 */
iftDblArray *iftCopyDblArray(const double* array, long n);


/**
 * @brief Destroys an iftDblArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
void iftDestroyDblArray(iftDblArray **darr);


/**
 * @brief Reallocates memory for an iftDblArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeDblArray(iftDblArray **iarr, long n);


/**
 * @brief Rearrange an array to a string for printing.
 * @author Samuka;
 * @date Sep 16, 2017
 */
char *iftDblArrayAsString(const iftDblArray *darr);


/**
 * @brief Creates an Array of Strings with at most 2048 characters.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
iftStrArray *iftCreateStrArray(long n);


/**
 * @brief Destroys an iftStrArray.
 * @author Samuel Martins
 * @date Oct 15, 2015
 * @ingroup Memory
 */
void iftDestroyStrArray(iftStrArray **sarr);


/**
 * @brief Reallocates memory for an iftStrArray and copies the original data. The new size could be higher or lower
 * @warning If the original array is larger then some data will be lost, i.e. only n elements will be copied
 * @author Cesar Castelo
 * @date Jul 18, 2018
 * @ingroup Memory
 */
void iftResizeStrArray(iftStrArray **iarr, long n);


/**
 * @brief Copies an iftStrArray.
 * @author Samuel Martins
 * @date Sep 17, 2017
 * @ingroup Memory
 */
iftStrArray *iftCopyStrArray(char **str_arr, long n);


/**
 * @brief Rearrange an array to a string for printing.
 * @author Samuka;
 * @date Sep 16, 2017
 */
char *iftStrArrayAsString(const iftStrArray *sarr);


/**
 * @brief Sets all values in <array_dst> to <val>.
 *
 * @param array_dst The array.
 * @param nelems The number of elements.
 * @param val value to set array
 * @ingroup Memory
 */
void iftSetFloatArray(float *array_dst, int nelems, float val);

/**
 * @brief Copies the float <array_src> to <array_dst>. We expect the number of elements to match.
 *
 * @param array_dst The destination array.
 * @param array_src The source array.
 * @param nelems The number of elements.
 * @ingroup Memory
 */
void iftCopyFloatArray(float *array_dst, float *array_src, int nelems);

/**
 * @brief Copies the float <array_src> to the double <array_dst>. We expect the number of elements to match.
 *
 * @param array_dst The destination double array.
 * @param array_src The source float array.
 * @param nelems The number of elements.
 * @ingroup Memory
 * 
 * @author Samuel Martins
 * @date May 15, 2018
 */
void iftCopyFloatArrayToDblArray(double *array_dst, float *array_src, int nelems);

/**
 * @brief Copies the double <array_src> to the float <array_dst>. We expect the number of elements to match.
 *
 * @param array_dst The destination float array.
 * @param array_src The source double array.
 * @param nelems The number of elements.
 * @ingroup Memory
 * 
 * @author Samuel Martins
 * @date May 15, 2018
 */
void iftCopyDblArrayToFloatArray(float *array_dst, double *array_src, int nelems);

/**
 * @brief Copies the double <array_src> to <array_dst>. We expect the number of elements to match.
 *
 * @param array_dst The destination array.
 * @param array_src The source array.
 * @param nelems The number of elements.
 * @ingroup Memory
 */
void iftCopyDoubleArray(double *array_dst, double *array_src, int nelems);
/**
 * @brief Copies the int <array_src> to <array_dst>. We expect the number of elements to match.
 *
 * @param array_dst The destination array.
 * @param array_src The source array.
 * @param nelems The number of elements.
 * @ingroup Memory
 */
void iftCopyIntArray(int *array_dst, const int *array_src, int nelems);
/**
 * @brief Concatenates two int arrays.
 *
 * @param array1 The first array.
 * @param n1 The number of elements in the first array.
 * @param array2 The first array.
 * @param n2 The number of elements in the second array.
 * @param nelems Returns the number of elements in the new array.
 * @return The concatenated arrays.
 */
int *iftConcatIntArray(int *array1, int n1, int *array2, int n2, int *nelems);

/**
 * @brief Sets all values in <array_dst> to <val>.
 *
 * @param array_dst The array.
 * @param nelems The number of elements.
 * @param val value to set array
 * @ingroup Memory
 */
void iftSetIntArray(int *array_dst, int nelems, int val);

/**
 * @author Peixinho
 * @date Nov, 2016
 * @brief Creates a double array in the specified range
 * @param begin Starting value
 * @param end Ending value (inclusive)
 * @param inc Increment ammount
 * @return
 */
iftDblArray *iftRange(double begin, double end, double inc);


/**
 * @brief Creates an integer array in the specified range.
 *
 * @note Inpired in range function from python.
 * 
 * @param begin Starting value
 * @param end Ending value (inclusive)
 * @param inc Increment ammount
 * @return Range Integer Array
 * 
 * @author Samuka
 * @date Jan 3, 2017
 */
iftIntArray *iftIntRange(int begin, int end, int inc);


/**
 * @brief Creates an integer array by repeating <n_repetitions> the value <x>.
 * 
 * @note The size of the integer array will be <n_repetitions>
 * 
 * @param x Value to be repeated.
 * @param n_repetitions Number of repetitions.
 * @return  Integer array with <n_repetitions> of the value <x>.
 * 
 * @author Samuka
 * @date Aug 23, 2018
 */
iftIntArray *iftIntRepeat(int x, int n_repetitions);


/**
 * @brief Creates a string array with specific values.
 * @author Peixinho
 * @param n Number of values to search.
 * @param ... Sequence of values to search.
 */
iftStrArray *iftStrValues(int n, ...);

/**
 * @brief Creates a double array with specific values.
 * @warning Since this function uses undefined number of arguments, you HAVE to pass double values as parameter, not integers.
 * @author Peixinho
 * @param n Number of values to search.
 * @param ... Sequence of values to search.
 */
iftDblArray *iftDblValues(int n, ...);

/**
 * @author Peixinho
 * @date Nov, 2016
 * @brief Creates a double array in the specified geometric range
 * @param begin Starting value
 * @param end Ending value (not inclusive)
 * @param inc Geommetric increment ammount
 * @return
 */
iftDblArray *iftGeomRange(double begin, double end, double mul);

void iftCopyData(void* destination,const void* source,long n, long elementSize);


void iftCopySizeTArray(long *array_dst, const long *array_src, long nelems);

/**
 * @author Cesar Castelo
 * @date Jan 23, 2018
 * @brief Initializes a float array with a given value
 * @param array Float array
 * @param value Value to initialize the array
 * @param nelems Size of the array
 * @return
 */
void iftInitFloatArray(float *array, float value, int nelems);

/**
 * @author Cesar Castelo
 * @date Jan 23, 2018
 * @brief Initializes a single position in a float array with a given value
 * @param array Float array
 * @param pos Position to be initialized
 * @param value Value to initialize the array
 * @param nelems Size of the array
 * @return
 */
void iftInitFloatArrayPos(float *array, int pos, float value, int nelems);

/**
 * @author Cesar Castelo
 * @date Jan 23, 2018
 * @brief Adds two float arrays and save the result in a third array
 * @param array1 Float array to be added
 * @param array2 Float array to be added
 * @param array3 Float array to store the result
 * @param nelems Size of the array
 * @return
 */
void iftSumFloatArrays(float *array1, float *array2, float *array3, int nelems);


/**
 * @brief Sums all elements from the float array <farr>.
 * @author Samuel Martins
 * @date Aug 21, 2018
 */
float iftSumFloatArrayElems(float *farr, int n_elems);


/**
 * @author Cesar Castelo
 * @date Ago 21, 2018
 * @brief Adds two float arrays and save the result in the first array
 * @param array1 Float array to be added and to store the result
 * @param array2 Float array to be added
 * @param nelems Size of the array
 * @return
 */
void iftSumFloatArraysInPlace(float *array1, float *array2, int nelems);

/**
 * @author Cesar Castelo
 * @date Jan 23, 2018
 * @brief Scale (multiply) a float array by a given value
 * @param array Float array to be scaled
 * @param value Value to scale the array
 * @param arrayOut Float array to store the result
 * @param nelems Size of the array
 * @return
 */
void iftScaleFloatArray(float* array, float value, float* arrayOut, int nelems);

/**
 * @author Cesar Castelo
 * @date Feb 01, 2018
 * @brief Merges two int arrays and stores the result in the first array
 * @param array1 Array to be merged (which will store the result)
 * @param array2 Array to be merged
 * @param n1 Size of array1
 * @param n2 Size of array2
 * @return
 */
void iftMergeIntArraysInPlace(int **array1, int *array2, int n1, int n2);


/**
 * @brief Creates an array of Voxels.
 * 
 * @param n Number of elements.
 * @return Array of Voxels
 * 
 * @author Samuka Martins
 * @date Aug 23, 2018
 */
iftVoxelArray *iftCreateVoxelArray(long n);

/**
 * @brief Destroys an Voxel Array.
 * @author Samuka Martins
 * @date Aug 23, 2018
 */
void iftDestroyVoxelArray(iftVoxelArray **varr);

#endif //IFT_IFTMEMORY_H
