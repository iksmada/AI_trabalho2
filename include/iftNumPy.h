/**
 * @file iftNumPy.h
 * @brief Definitions and functions about an interface to NumPy files.
 * @author Samuel Martins
 * @date Nov 14, 2017
 *
 * Based on http://pyopengl.sourceforge.net/pydoc/numpy.lib.format.html
 */

#ifndef IFT_NUMPY_H
#define IFT_NUMPY_H

#ifdef __cplusplus
extern "C" {
#endif


#include "iftFImage.h"
#include "iftImage.h"
#include "iftMemory.h"



/**
 * @brief Read a 1-D Numpy array as an Integer Array.
 * @param  npy_path Pathname from the Numpy array file.
 * @return          1-D Integer Array.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
iftIntArray *iftReadNumPyAsIntArray(const char *npy_path);

/**
 * @brief Read a 1-D Numpy array as a Float Array.
 * @param  npy_path Pathname from the Numpy array file.
 * @return          1-D Float Array.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
iftFloatArray *iftReadNumPyAsFloatArray(const char *npy_path);

/**
 * @brief Read a 2-D Numpy array as an Integer Matrix.
 * @param  npy_path Pathname from the Numpy array file.
 * @return          2-D Integer Matrix.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
iftIntMatrix *iftReadNumPyAsIntMatrix(const char *npy_path);

/**
 * @brief Read a 2-D Numpy array as a Float Matrix.
 * @param  npy_path Pathname from the Numpy array file.
 * @return          2-D Float Matrix.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
iftMatrix *iftReadNumPyAsMatrix(const char *npy_path);

/**
 * @brief Read a Numpy array as an Image.
 * @param  npy_path Pathname from the Numpy array file.
 * @return          Image.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
iftImage *iftReadNumPyAsImage(const char *npy_path);

/**
 * @brief Read a Numpy array as a Float Image.
 * @param  npy_path Pathname from the Numpy array file.
 * @return          Float Image.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
iftFImage *iftReadNumPyAsFImage(const char *npy_path);


/**
 * @brief Write an Integer Array as a 1-D Numpy array.
 * @param  arr      Integer Array.
 * @param  npy_path Pathname from the Numpy array file.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
void iftWriteIntArrayAsNumPy(const iftIntArray *arr, const char *npy_path);

/**
 * @brief Write a Float Array as a 1-D Numpy array.
 * @param  arr      Float Array.
 * @param  npy_path Pathname from the Numpy array file.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
void iftWriteFloatArrayAsNumPy(const iftFloatArray *arr, const char *npy_path);

/**
 * @brief Write an Integer Matrix as a 2-D Numpy array.
 * @param  mat      Integer Matrix.
 * @param  npy_path Pathname from the Numpy array file.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
void iftWriteIntMatrixAsNumPy(const iftIntMatrix *mat, const char *npy_path);

/**
 * @brief Write a Float Matrix as a 2-D Numpy array.
 * @param  mat      Integer Matrix.
 * @param  npy_path Pathname from the Numpy array file.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
void iftWriteMatrixAsNumPy(const iftMatrix *mat, const char *npy_path);

/**
 * @brief Write an Image as a Numpy array.
 * @param  img      Image.
 * @param  npy_path Pathname from the Numpy array file.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
void iftWriteImageAsNumPy(const iftImage *img, const char *npy_path);

/**
 * @brief Write a Float Image as a Numpy array.
 * @param  fimg     Float Image.
 * @param  npy_path Pathname from the Numpy array file.
 * 
 * @author Samuel Martins
 * @date Nov 14, 2017
 */
void iftWriteFImageAsNumPy(const iftFImage *fimg, const char *npy_path);


#ifdef __cplusplus
}
#endif

#endif












