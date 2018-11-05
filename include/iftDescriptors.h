#ifndef IFT_DESCRIPTORS_H_
#define IFT_DESCRIPTORS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/stat.h>

#include "iftCommon.h"
#include "iftImage.h"
#include "iftMImage.h"
#include "iftAdjacency.h"
#include "iftRadiometric.h"

/**
 * @brief Feature vector
 * @details [long description]
 * @author Adan Echemendia
 * @date Jun 10, 2016
 */
typedef struct ift_features {

    /** List of feature values */
    float *val;
    /** Size of the feature vector */
    int    n;
} iftFeatures;

typedef struct ift_hog {
    iftVoxel cellSize;
    iftVoxel blockSize;
    iftVoxel blockStride;
    int nbins;
} iftHog;

/**
 * @brief Creates a feature vector
 *
 * @param n Size of the feature vector
 * @return A feature vector
 */
iftFeatures *iftCreateFeatures(int n);

/**
 * @brief Destroys a feature vector
 *
 * @param feat Pointer to pointer to a feature vector
 */
void         iftDestroyFeatures(iftFeatures **feat);

/**
 * @brief Reads a set of features from a file and builds a feature vector
 * @details [long description]
 *
 * @param filename File path
 * @return A feature vector
 */
iftFeatures *iftReadFeatures(const char *filename);

/**
 * @brief Writes a feature vector to a file
 *
 * @param feats Feature vector
 * @param filename File path
 */
void         iftWriteFeatures(const iftFeatures *feats, const char *filename);

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param img [description]
 * @param A [description]
 *
 * @return [description]
 */
iftImage *iftLocalBinaryPattern(const iftImage *img, const iftAdjRel *A);

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param img [description]
 * @param A [description]
 *
 * @return [description]
 */
iftFeatures *iftExtractLBP(iftImage *img, iftAdjRel *A);


/**
 * @brief Codes a 3D-LBP from a 3D image.
 * 
 * Codes a 3D-LBP from a 3D image by using the algorithm LBP-TOP proposed in [1].
 * For each orientation (XY, ZX, YX), it computes a 2D LBP for each corresponding slice into
 * a 3D image.
 * We use 8-neighborhood to compute the 2D LBP values.
 * The function returns an array with 3 images that will be used to extract the feature vectors (@see iftExtract3DLBPTOPFeats)
 * [0] = 3D LBP (XY)
 * [1] = 3D LBP (ZX)
 * [2] = 3D LBP (YZ)
 * 
 * [1] Zhao, Guoying, and Matti Pietikainen. "Dynamic texture recognition using local binary patterns
 * with an application to facial expressions." IEEE transactions on pattern analysis and
 * machine intelligence 29.6 (2007): 915-928.
 * 
 * @param img Input 3D Image.
 * @return Returned array of 3D-LBP images.
 */
iftImageArray *ift3DLBPTOP(const iftImage *img);


/**
 * @brief Extracts the LBP features from an input 3D Image.
 * 
 * This function is based on the algorithm LBP-TOP proposed in [1].
 * Given the 3 LBP images, one for each orientation, coded by the function @see ift3DLBPTOP,
 * the resulting feature vector is the concatenation of the histogram of each LBP image.
 * 
 * If a mask is passed, the histograms are only computed inside it.
 * 
 * @param img Input 3D image.
 * @param mask (Optional) Mask that indicates the voxels to be considered during feature extraction.
 * @return Resulting LBP feature vector.
 * 
 * @author Samuel Martins
 * @date Aug 27, 2018
 */
iftFeatures *iftExtract3DLBPTOPFeats(const iftImage *img, const iftImage *mask, bool normalize_histograms);


/**
 * @brief Codes a 3D-LBP from a 3D image using the method VLBP proposed in [1].
 * 
 * [1] Zhao, Guoying, and Matti Pietikainen. "Dynamic texture recognition using local binary patterns
 * with an application to facial expressions." IEEE transactions on pattern analysis and
 * machine intelligence 29.6 (2007): 915-928.
 * 
 * @param img Input 3D image.
 * @return Resulting LBP image.
 * 
 * @author Samuel Martins
 * @date Aug 28, 2018
 */
iftImage *iftVLBP(const iftImage *img);


/**
 * @brief Extracts the LBP features from an input 3D Image using the method VLBP proposed in [1].
 * 
 * [1] Zhao, Guoying, and Matti Pietikainen. "Dynamic texture recognition using local binary patterns
 * with an application to facial expressions." IEEE transactions on pattern analysis and
 * machine intelligence 29.6 (2007): 915-928.
 * 
 * @param img Input 3D image.
 * @param mask (Optional) Mask that indicates the voxels to be considered during feature extraction.
 * @param nbins Number of bins of the LBP histogram. If <= 0, use the default number 2ˆ14.
 * @param normalize_hist Normalize the LBP histogram.
 * @return Resulting LBP feature vector.
 * 
 * @author Samuel Martins
 * @date Aug 28, 2018
 */
iftFeatures *iftExtractVLBPFeats(const iftImage *img, const iftImage *mask, int nbins,
                                 bool normalize_hist);


/**
 * @brief Builds a feature vector with the normalized brightness values of an image
 * @details [long description]
 *
 * @param img Target image
 * @return Feature vector with the normalized brightness values of an image
 */
iftFeatures *iftExtractBrightness(iftImage *img);

/**
 * @brief Builds a feature vector with the normalized brightness values of the gradient magnitude image of image <br>img</br>
 * @details [long description]
 *
 * @param img Target image
 * @param A Adjacency relation
 *
 * @return Feature vector with the normalized brightness values of the gradient magnitude image of image <br>img</br>
 */
iftFeatures *iftExtractGradient(iftImage *img, iftAdjRel *A);

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param img [description]
 * @param bins_per_band [description]
 *
 * @return [description]
 */
iftFeatures *iftExtractBIC(iftImage *img, int bins_per_band);

/**
 * @brief extract BIC features performing the quantization process by clustering in 1D (e.g. a grayscale image)
 * @author Cesar Castelo
 * @date Nov 28, 2017
 * @details
 *
 * @param img input image in the YCbCr color space
 * @param bins_per_band number of bins per band
 *
 * @return [description]
 */
 iftFeatures *iftExtractBICClustering1D(iftImage *img, int bins_per_band);

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param img [description]
 * @param mask [description]
 * @param bins_per_band [description]
 * @return [description]
 */
iftFeatures *iftExtractBICInMask(iftImage *img, iftImage *mask, int bins_per_band);

/**
 * @brief Computes the Manhattan distance between two feature vectors
 * @details [long description]
 *
 * @param feat1 Feature vector one
 * @param feat2 Feature vector two
 *
 * @return Manhattan distance value between the two feature vectors
 */
float        iftFeatDistL1(iftFeatures *feat1, iftFeatures *feat2);

/**
 * @brief Builds a feature vector with all values of a multi-band image
 * @details [long description]
 *
 * @param img Target image
 * @return Feature vector with all values of a multi-band image
 */
iftFeatures *iftMImageToFeatures(iftMImage *img);

/**
 * @brief Writes a feature vector to a file. The first saved value is the true label of the sample.
 * @details [long description]
 *
 * @param features Feature vector of the sample
 * @param truelabel True label of the sample
 * @param fp File path
 */
void     iftWriteFeaturesInFile(iftFeatures *features, int truelabel, FILE *fp);

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param img [description]
 * @param u1 [description]
 * @param u2 [description]
 * @return [description]
 */
iftFeatures *iftIntensityProfile(iftImage *img, iftPoint u1, iftPoint u2);

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param img [description]
 * @param bins_per_band [description]
 * @param normalization_value [description]
 * @return [description]
 */
iftImage    *iftQuantize(iftImage *img, int bins_per_band, int normalization_value);

/**
 * @brief quantize image by clustering in 1D (e.g. for a grayscale image)
 * @details [long description]
 * @author Cesar Castelo
 * @date Nov 28, 2017
 *
 * @param img input image
 * @param bins_per_band number of bins per band
 * @param normalization_value normalization value
 * @param knn_create_method method to be used to create the knn graph for OPF clustering
 * @return [description]
 */
iftImage    *iftQuantizeByClustering1D(iftImage *img, int bins_per_band);

/**
 * @brief Creates and defines the HOG feature extractor parameters.
 * @param cellSize The cell size considered in the histogram computation.
 * @param blockSize The block size defines the number of cells to be considered in the normalization.
 * @param blockStride The block stride defines the overlap between blocks in the normalization.
 * @author Peixinho
 * @date April, 2016
 */
iftHog* iftCreateHog2D(int cellSize, int blockSize, int blockStride, int nbins);

/**
 * @brief Destroys a HOG
 *
 * @param Pointer to pointer to a HOG
 */
void iftDestroyHog(iftHog** hog);

/**
 * @brief Extracts the Hog descriptor of an image.
 * @author Peixinho
 * @date April, 2016
 */
iftFeatures* iftExtractHOG2D(const iftHog *hog, iftImage *img);

#ifdef __cplusplus
}
#endif

#endif
