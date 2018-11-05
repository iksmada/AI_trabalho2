/**
 * @file iftBagOfVisualWords.h
 * @brief Bag of Visual Words module
 * @author Cesar Castelo
 * @date  January 4, 2018
 */

#ifndef IFT_IFTBAGOFVISUALWORDS_H
#define IFT_IFTBAGOFVISUALWORDS_H

#include "stdarg.h"
#include "iftDataSet.h"
#include "iftKmeans.h"
#include "iftMatrix.h"
#include "iftMetrics.h"
#include "iftDescriptors.h"
#include "iftFile.h"
#include "iftDialog.h"
#include "iftBoundingBox.h"
#include "iftColor.h"
#include "iftClustering.h"
#include "iftBasicDataTypes.h"
#include "iftGraphics.h"

#define BOVW_INT_POINT_DETECTOR_RANDOM 0
#define BOVW_INT_POINT_DETECTOR_DENSE 1
#define BOVW_INT_POINT_DETECTOR_SUPVOX_BOUNDARY 2
#define BOVW_INT_POINT_DETECTOR_SUPVOX_CENTER 3

#define BOVW_LOCAL_FEAT_EXTRACTOR_RAW 0
#define BOVW_LOCAL_FEAT_EXTRACTOR_BIC 1
#define BOVW_LOCAL_FEAT_EXTRACTOR_LBP 2

#define BOVW_DICT_ESTIMATOR_UNSUP_KMEANS 0
#define BOVW_DICT_ESTIMATOR_SUP_PIX_KMEANS 1
#define BOVW_DICT_ESTIMATOR_SUP_IMG_KMEANS 2
#define BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_KMEANS 3
#define BOVW_DICT_ESTIMATOR_UNSUP_OPF 4
#define BOVW_DICT_ESTIMATOR_SUP_PIX_OPF 5
#define BOVW_DICT_ESTIMATOR_SUP_IMG_OPF 6
#define BOVW_DICT_ESTIMATOR_SUP_PIX_AND_IMG_OPF 7

#define BOVW_DICT_ESTIMATOR_CLASS_HRCHY_PIX 0
#define BOVW_DICT_ESTIMATOR_CLASS_HRCHY_IMG 1

#define BOVW_COD_FUNC_HARD_ASGMT 0
#define BOVW_COD_FUNC_SOFT_ASGMT 1

#define BOVW_DIST_FUNC_1 1
#define BOVW_DIST_FUNC_2 2
#define BOVW_DIST_FUNC_3 3
#define BOVW_DIST_FUNC_4 4
#define BOVW_DIST_FUNC_5 5
#define BOVW_DIST_FUNC_6 6
#define BOVW_DIST_FUNC_7 7
#define BOVW_DIST_FUNC_8 8
#define BOVW_DIST_FUNC_9 9
#define BOVW_DIST_FUNC_10 10
#define BOVW_DIST_FUNC_11 11
#define BOVW_DIST_FUNC_12 12

#define BOVW_POOL_FUNC_SUM 0
#define BOVW_POOL_FUNC_MAX 1

#define BOVW_ACTIV_FUNC_ALL 0

#define BOVW_NORM_TYPE_NONE 0

#define BOVW_COD_FUNC_SOFT_ASGMT_WGT_FUNC_INVER 0
#define BOVW_COD_FUNC_SOFT_ASGMT_WGT_FUNC_GAUSS 1

/* ---------------------------------------------------------------------------------------------------------------------*/
/*                                                Structure Definition                                                  */
/* ---------------------------------------------------------------------------------------------------------------------*/

/** @addtogroup Descriptor
 * @{ */

/** @addtogroup BagOfVisualWords
 * @brief Bag of Visual Words descriptor
 * @{ */

/** @brief Interest Points Detector Prototype
 * Interface for interest points detector
 * @param img Image to select samples
 * @param imgLabel pixel-level labels of img (mask)
 * @param imgFilePtr Pointer to the image file
 * @param saveImgIntPts Whether or not to save the image with the interest points
 * @param funcParams Extra parameters to be received (for the instanced function)
 * @return iftBoundingBoxArray that represents the regions of interest
 * @author Cesar Castelo
 * @date Jan 04, 2018
 **/
typedef iftBoundingBoxArray* (*iftBovwIntPointDetector)(iftImage* img, iftImage* imgLabel, iftFile *imgFilePtr, bool saveImgIntPts, iftDict *funcParams);

/**
 * @brief Local Features Extractor Prototype
 * Interface for local feature extractor
 * @param img Image to extract patch features
 * @param roi Regions of interest
 * @param imgFilePtr Pointer to the image file
 * @param patchDirPath Directory to save the image patches that will be created (reference data). If NULL the reference data will not be created
 * @param funcParams Extra parameters to be received (for the instanced function)
 * @return iftDataSet that contains the local features
 * @author Cesar Castelo
 * @date Jan 04, 2018
 **/
typedef iftDataSet* (*iftBovwLocalFeatExtractor)(iftImage* img, iftBoundingBoxArray* roi, iftFile *imgFilePtr, char *patchDirPath, iftDict *funcParams);

/**
 * @brief Dictionary Estimator Prototype.
 * Interface to dictionary estimator
 * @param localFeats iftDataset containing the set of local features extracted from a set of images
 * @param funcParams Extra parameters to be received (for the instanced function)
 * @return iftMatrix that contains the dictionary (bag of visual words)
 * @author Cesar Castelo
 * @date Jan 04, 2018
 **/
typedef iftMatrix* (*iftBovwDictEstimator)(iftDataSet* localFeats, iftDict *funcParams);

/**
 * @brief Coding Function Prototype
 * Interface to coding functions for a new image
 * @param dict iftMatrix representing the visual dictionary
 * @param localFeats Set of local features extracted from the image to be coded
 * @param funcParams Extra parameters to be received (for the instanced function)
 * @return iftFeatures that represents the activations for each regionOfInterest against every visual word on the dictionary
 * @author Cesar Castelo
 * @date Jan 04, 2018
 **/
typedef iftFeatures* (*iftBovwCodFunc)(iftMatrix* dict, iftDataSet* localFeats, iftDict *funcParams);

/**
 * @brief Pooling Function Prototype
 * Interface to pooling functions for a new image (previously coded).
 * @param activations iftMatrix representing the visual dictionary
 * @param norm Normalization to be used
 * @param nParams Number of extra parameters to be received (for the instanced function)
 * @param ... Parameters (different for each function instance)
 * @return iftFeatures that represent the final feature vector
 * @author Cesar Castelo
 * @date Jan 04, 2018

typedef iftFeatures* (*iftBovwPoolFunc)(iftMatrix* activations, int norm, int nParams, ...);
 **/

/**
 * @brief Bag of Visual Words Structure
 * Contains the BoVW architecture information along with the visual dictionary learned.
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
typedef struct ift_bag_of_visual_words
{
    /* set of local features extracted from all the images */
    iftDataSet *localFeats;
    int numbLocalFeats; // number of local features extracted for all images
    int dimLocalFeats; // dimensionality of the local feature vectors extracted in each image patch

    /* dictionary (visual words) */
    iftMatrix *dict;
    int dictSize; // dictionary size (number of visual words)

    /* class-hierarchy-related properties */
    iftIntMatrix *localFeatsClassHrchy; // label hierarchy for the local feats from all images
    iftIntArray *classHrchyIds; // Ids of the class hierarchies
    int numbClassHrchies; // number of class hierarchies in the problem

    /* path to images used to create the dictionary */
    iftFileSet *trainImgFileSet;
    iftFileSet *trainImgLabelFileSet;

    /* Function to detect interest points in the image */
    iftBovwIntPointDetector intPointDetector;
    int intPointDetectorId;

    /* Function to extract local features from the interest point detected  */
    iftBovwLocalFeatExtractor localFeatExtractor;
    int localFeatExtractorId;

    /* Function to estimate the visual dictionary */
    iftBovwDictEstimator dictEstimator;
    int dictEstimatorId;

    /* Function to apply the coding operation */
    iftBovwCodFunc codFunc;
    int codFuncId;

    /* distance function used to compute the dictionary and to perform the coding operation */
    iftArcWeightFun arcWeightFunc;
    int distFuncId;

    /* function pointers' params */
    iftDict *funcParams;

} iftBagOfVisualWords;

/* ---------------------------------------------------------------------------------------------------------------------*/
/*                                                Initialization functions                                              */
/* ---------------------------------------------------------------------------------------------------------------------*/

/**
 * @brief Creates a BoVW structure.
 * @return The Bag of Visual Words structure
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftBagOfVisualWords *iftCreateBovw();

/**
 * @brief Sets the function pointers according to the chosen parameters
 * @param bovw BoVW structure
 * @param intPointDetector Function to detect interest points in the image
 * @param localFeatExtractor Function to extract local features from the interest point detected
 * @param dictEstimator Function to estimate the visual dictionary
 * @param codFunc Function to apply the coding operation
 * @param distFunc Distance function to be used
 * @param funcParams Parameters for the function pointers
 * @author Cesar Castelo
 * @date Jan 30, 2018
 */
void iftBovwSetFunctionPointers(iftBagOfVisualWords *bovw, int intPointDetector, int localFeatExtractor, int dictEstimator, int codFunc, int distFunc, iftDict *funcParams);

/**
 * @brief Destroys a BoVW structure.
 * @param bovw BoVW structure to be destroyed
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
void iftDestroyBovw(iftBagOfVisualWords **bovw);

/**
 * @brief Saves a BoVW structure.
 * @param bovw BoVW structure to be saved
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
void iftBovwWrite(iftBagOfVisualWords *bovw);

/**
 * @brief Reads a BoVW structure.
 * @param fileName File that contains the BoVW structure
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftBagOfVisualWords *iftBovwRead(const char *filename);

/* ---------------------------------------------------------------------------------------------------------------------*/
/*                                                Learning functions                                               */
/* ---------------------------------------------------------------------------------------------------------------------*/

/**
 * @brief Learns a Bag of Visual Words structure for image classification
 * @param bovw Bag of Visual Words structure to be trained,
 * @param trainImgFileSet image set that will be used to train the dictionary
 * @param trainImgLabelFileSet image set that contains the pixel labels (mask) of trainImgFileSet
 * @param saveImgIntPts Whether or not to save the image with the interest points
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
void iftBovwLearnImgClassif(iftBagOfVisualWords *bovw, iftFileSet *trainImgFileSet, iftFileSet *trainImgLabelFileSet, bool saveImgIntPts);

/**
 * @brief Detect interest points and extract local features from an image (using the function pointers)
 * @param bovw Bag of Visual Words structure
 * @param img Input image
 * @param trainImgFileSet Training image fileset
 * @param fileId File Id of the image inside trainImgFileSet (for reference data)
 * @author Cesar Castelo
 * @date Feb 06, 2018
 */
//iftDataSet* iftBovwDetectAndExtractLocalFeatsFromImage(iftBagOfVisualWords *bovw, iftImage *img, iftFileSet *trainImgFileSet, int fileId);

/**
 * @brief Computes the feature vector for an image using a previously trained dictionary (it uses the coding and pooling functions)
 * @param bovw Bag of Visual Words structure
 * @param img Image to extract the feature vector from
 * @param imgLabel pixel-level labels of img (mask)
 * @param imgFilePtr Pointer to the image file
 * @param saveImgIntPts Whether or not to save the image with the interest points
 * @return The computed feature vector
 * @author Cesar Castelo
 * @date Feb 08, 2018
 */
iftFeatures *iftBovwComputeFeatVectFromImage(iftBagOfVisualWords *bovw, iftImage *img, iftImage*imgLabel, iftFile *imgFilePtr, bool saveImgIntPts);

/**
 * @brief Computes the feature vector for an image fileset using a previously trained dictionary
 * It calls the iftBovwComputeFeatVectFromImage function for every image
 * @param bovw Bag of Visual Words structure
 * @param imgFileSet Image fileset to extract the feature vectors from
 * @param imgLabelFileSet fileset containing the pixel-levels of imgFileSet
 * @param saveImgIntPts Whether or not to save the image with the interest points
 * @return A dataset containing the feature vectors
 * @author Cesar Castelo
 * @date Feb 08, 2018
 */
iftDataSet *iftBovwComputeFeatVectsFromImgFileSet(iftBagOfVisualWords *bovw, iftFileSet *imgFileSet, iftFileSet *imgLabelFileSet, bool saveImgIntPts);

/**
 * @brief Computes the vector of local features for an image fileset
 * @param bovw Bag of Visual Words structure
 * @param trainImgFileSet Image Fileset to extract the feature vectors from
 * @param trainImgLabelFileSet Fileset containing the pixel-level labels of trainImgFileSet
 * @param patchDirPath Directory to save the image patches that will be created (reference data)
 * @param saveImgIntPts Whether or not to save the image with the interest points
 * @return A dataset containing the local feature vectors
 * @author Cesar Castelo
 * @date Apr 20, 2018
 */
void iftComputeLocalFeatsFromImgFileSet(iftBagOfVisualWords *bovw, iftFileSet *trainImgFileSet, iftFileSet *trainImgLabelFileSet, char *patchDirPath, bool saveImgIntPts);

// iftMImage* iftBowTransform(iftBagOfFeatures *bow, iftImage *img); // Compute Bow multiband image from image.

/* ---------------------------------------------------------------------------------------------------------------------*/
/*                                            Function pointers Implementation                                          */
/* ---------------------------------------------------------------------------------------------------------------------*/

/**
 * @brief Detects random interest points from image
 * @implements intPointDetector()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -nPoints: Number of points (int)
 * -patchSize: Size of each patch (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftBoundingBoxArray* iftBovwRandomIntPointDetector(iftImage* img, iftImage* imgLabel, iftFile *imgFilePtr, bool saveImgIntPts, iftDict *funcParams);

/**
 * Detects interest points from image in a dense way
 * @implements intPointDetector()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -patchSize: Size of each patch (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftBoundingBoxArray* iftBovwDenseIntPointDetector(iftImage* img, iftImage* imgLabel, iftFile *imgFilePtr, bool saveImgIntPts, iftDict *funcParams);

/**
 * Divides the images in supervoxels and set them as regions of interest (minimum bounding box)
 * @implements intPointDetector()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -nSupervoxels: Number of supervoxels (int)
 *
 * @author Cesar Castelo
 * @date Abr 27, 2018
 */
iftBoundingBoxArray* iftBovwSupervoxelCenterIntPointDetector(iftImage* img, iftImage* imgLabel, iftFile *imgFilePtr, bool saveImgIntPts, iftDict *funcParams);

/**
 * Divides the images in supervoxels and take points from the supervoxels' boundaries
 * @implements intPointDetector()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -nSupervoxels: Number of supervoxels (int)
 * -patchSize: Size of each patch (int)
 *
 * @author Cesar Castelo
 * @date Abr 27, 2018
 */
iftBoundingBoxArray* iftBovwSupervoxelBoundaryIntPointDetector(iftImage* img, iftImage* imgLabel, iftFile *imgFilePtr, bool saveImgIntPts, iftDict *funcParams);

/**
 * @brief Extracts the Raw pixels color/brightness information from a set of regions of interest
 * @implements iftBovwLocalFeatExtractor()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -colSpaceIn: Input color space (iftColorSpace)
 * -colSpaceOut: Output color space (iftColorSpace)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftDataSet* iftBovwRawFeatExtractor(iftImage* img, iftBoundingBoxArray* roi, iftFile *imgFilePtr, char *patchDirPath, iftDict *funcParams);

/**
 * @brief Extract BIC features from a set of regions of interest
 * @implements iftBovwLocalFeatExtractor()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -nBins: Number of bins for image quantization (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftDataSet* iftBovwBICFeatExtractor(iftImage* img, iftBoundingBoxArray* roi, iftFile *imgFilePtr, char *patchDirPath, iftDict *funcParams);

/**
 * @brief Extract LBP features from a set of regions of interest
 * @implements iftBovwLocalFeatExtractor()
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftDataSet* iftBovwLBPFeatExtractor(iftImage* img, iftBoundingBoxArray* roi, iftFile *imgFilePtr, char *patchDirPath, iftDict *funcParams);

/**
 * @brief Unsupervised KMeans dictionary estimator for BoVW
 * @implements iftBovwDictEstimator()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -retRealCent: Whether to use the real centroids (present in the data) or the mean vector of each group (int)
 * -clustSampPerc: Percentage of the dataset to be used to compute the clusters (double)
 * -nGroups: Number of groups (int)
 * -maxIter: Maximum number of iterations (int)
 * -minImprov: Minimum improvement before stopping the k-means algorithm (double)
 * -decrGroups: Whether or not to decrement the resulting group numbers (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftMatrix* iftBovwUnsupKMeansDictEstimator(iftDataSet* localFeats, iftDict *funcParams);

/**
 * @brief One-level class-hierarchy supervised KMeans dictionary estimator for BoVW (e.g. pixel-level, image-level, etc)
 * It computes a KMeans dictionary estimator for each class at a certain one-level class-hierarchy, e.g., object and
 * background (for pixel-level), image category (for image-level), etc.
 * @implements iftBovwDictEstimator()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -retRealCent: Whether to use the real centroids (present in the data) or the mean vector of each group (int)
 * -clustSampPerc: Percentage of the dataset to be used to compute the clusters (double)
 * -nGroups: Number of groups (int)
 * -maxIter: Maximum number of iterations (int)
 * -minImprov: Minimum improvement before stopping the k-means algorithm (double)
 * -decrGroups: Whether or not to decrement the resulting group numbers (int)
 * -localFeatsClassHrchy: Class hierarchy for all the local features (iftIntMatrix*)
 *
 * @author Cesar Castelo
 * @date Feb 02, 2018
 */
iftMatrix* iftBovwSupOneLevelClassHrchyKMeansDictEstimator(iftDataSet* localFeats, iftDict *funcParams);

iftMatrix* iftBovwSupTwoLevelClassHrchyKMeansDictEstimator(iftDataSet* localFeats, iftDict *funcParams /*int retRealCent, double clustSampPerc, int nGroups, int maxIter, double minImprov, int decrGroups, iftIntMatrix *localFeatsClassHrchy*/ );

/**
 * @brief Unsupervised OPF dictionary estimator for BoVW
 * @implements iftBovwDictEstimator()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -retRealCent: Whether to use the real centroids (present in the data) or the mean vector of each group (int)
 * -clustSampPerc: Percentage of the dataset to be used to compute the clusters (double)
 * -knnGraphNeighbPerc: Percentage to be used to mount the knn graph, after clustSampPerc (double)
 * -knownNumbGroups: Whether or not we know the number of groups (int)
 * -nGroups: Number of groups (int)
 * -decrGroups: Whether or not to decrement the resulting group numbers (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftMatrix* iftBovwUnsupOPFDictEstimator(iftDataSet* localFeats, iftDict *funcParams);

/**
 * @brief One-level class-hierarchy supervised OPF dictionary estimator for BoVW (e.g. pixel-level, image-level, etc)
 * It computes an OPF dictionary estimator for each class at a certain one-level class-hierarchy, e.g., object and
 * background (for pixel-level), image category (for image-level), etc.
 * @implements iftBovwDictEstimator()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -retRealCent: Whether to use the real centroids (present in the data) or the mean vector of each group (int)
 * -clustSampPerc: Percentage of the dataset to be used to compute the clusters (double)
 * -knnGraphNeighbPerc: Percentage to be used to mount the knn graph, after clustSampPerc (double)
 * -knownNumbGroups: Whether or not we know the number of groups (int)
 * -nGroups: Number of groups (int)
 * -decrGroups: Whether or not to decrement the resulting group numbers (int)
 * -localFeatsClassHrchy: Class hierarchy for all the local features (iftIntMatrix*)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftMatrix* iftBovwSupOneLevelClassHrchyOPFDictEstimator(iftDataSet* localFeats, iftDict *funcParams);

iftMatrix* iftBovwSupTwoLevelClassHrchyOPFDictEstimator(iftDataSet* localFeats, iftDict *funcParams /*int retRealCent, double clustSampPerc, double knnGraphNeighbPerc, int knownNumbGroups, int nGroups, int decrGroups, iftIntMatrix *localFeatsClassHrchy*/ );

/**
 * @brief Hard clustering assignment function
 * @implements iftBovwCodFunc()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -simFunc: Similarity function to be used for the coding process (int)
 * -normType: Normalization to be used (int)
 * -poolFunc: Pooling function to be used for dictionary size reduction (int)
 * -activFunc: Activation function to be used (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftFeatures* iftBovwHardAsgmtCodFunc(iftMatrix* dict, iftDataSet* localFeats, iftDict *funcParams /* */);

/**
 * @brief Soft clustering assignment function
 * @implements iftBovwCodFunc()
 * @warning The parameters that should be sent to this function in funcParams are:
 * -simFunc: Similarity function to be used for the coding process (int)
 * -normType: Normalization to be used (int)
 * -poolFunc: Pooling function to be used for dictionary size reduction (int)
 * -activFunc: Activation function to be used (int)
 * -numbNearNeigh: Number of neighbors to be used for the soft assignment (int)
 * -wgtFunc: Weight function to be used for the soft assignment (int)
 *
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
iftFeatures* iftBovwSoftAsgmtCodFunc(iftMatrix* dict, iftDataSet* localFeats, iftDict *funcParams /* int , int  */);

//int iftBowHVSumPool(iftMImage *mImg, float *featsOut, int n); // Horizontal and vertical sum of intensities. Preserves spatial information.

/**
 * @brief Sum pooling function (to be applied after the coding operation)
 * @implements iftBovwPoolFunc()
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
//iftFeatures* iftBovwSumPoolFunc(iftMatrix* activations, int norm, int nParams, ...);

/**
 * @brief Max pooling function (to be applied after the coding operation)
 * @implements iftBovwPoolFunc()
 * @author Cesar Castelo
 * @date Jan 04, 2018
 */
//iftFeatures* iftBovwMaxPoolFunc(iftMatrix* activations, int norm, int nParams, ...);

/* ---------------------------------------------------------------------------------------------------------------------*/
/*                                                Complementary functions                                               */
/* ---------------------------------------------------------------------------------------------------------------------*/

/**
 * @brief Create the fileset reference data for a given set of regions of interest.
 * It returns the fileset and also saves the image patches in the given patchDirPath
 * @param img Image to extract the patches to be saved
 * @param roi Regions of interest inside the image
 * @param imgFilePtr Pointer to the image file
 * @param patchDirPath Directory where the image patches will be saved
 * @author Cesar Castelo
 * @date Abr 23, 2018
 */
iftFileSet *iftBoVWCreateFileSetRefDataFromImgROIs(iftImage *img, iftBoundingBoxArray *roi, iftFile *imgFilePtr, char *patchDirPath);

/**
 * @brief Compute the number of groups per class combination for a specific hierarchy
 * @param localFeatsClassHrchy Integer matrix that contains the entire class hierarchy (one or more levels)
 * @param nSamples Number of samples in the dataset
 * @param nGroups Total number of groups
 * @author Cesar Castelo
 * @date Feb 02, 2018
 */
void* iftBovwComputeNumbGroupsPerClassWithOneLevelHrchy(iftIntMatrix *localFeatsClassHrchy, int nSamples, int nGroups);

/**
 * @brief Compute the array with the labels corresponding to a specific hierarchy
 * @param feats Feature vectors representing the image patches
 * @param hrchyId Hierarchy Id
 * @param nParams Number of variable parameters
 * @param ... Variable parameters
 * @author Cesar Castelo
 * @date Feb 01, 2018
 */
int *iftBovwComputeClassHrchyArray(iftDataSet *feats, int hrchyId, int nParams, ...);

/**
 * @brief Sample a dataset based on the given label hierarchy
 * @param Z Dataset to be sampled
 * @param dataSetClassHrchy Matrix that contains the label hierarchy for the dataset
 * @param chosenLabels The specific combination of labels that will be used for the sampling
 * @date Feb 02, 2018
 * @author Cesar Castelo
 */
iftDataSet *iftBovwSampleDataSetByClassHrchy(iftDataSet *Z, iftIntMatrix *localFeatsClassHrchy, int *chosenLabels);

/**
 * @brief Prints messages regarding the chosen methods for each part of BoVW
 * @param intPointDetec Chosen interest point detection method (IFT_NIL to not print)
 * @param localFeatExtr Chosen local features extraction method (IFT_NIL to not print)
 * @param dictEstim Chosen dictionary estimation method (IFT_NIL to not print)
 * @param codFunc Chosen coding method (IFT_NIL to not print)
 * @date Feb 21, 2018
 * @author Cesar Castelo
 */
void iftBovwPrintChosenMethods(int intPointDetec, int localFeatExtr, int dictEstim, int codFunc);

/**
 * @brief Saves an image with the interest point detected from it
 * @param img Original image
 * @param roi Set of interest points to be added to the image
 * @param imgFilePtr Pointer to the image file
 * @date July 16, 2018
 * @author Cesar Castelo
 */
void iftBovwSaveImgWithIntPoints(iftImage *img, iftBoundingBoxArray *roi, iftFile *imgFilePtr);

/** @} */

/** @} */

#endif //IFT_IFTBAGOFVISUALWORDS_H
