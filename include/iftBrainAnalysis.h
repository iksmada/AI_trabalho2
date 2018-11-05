/**
 * @file iftBrainAnalysis.h
 * @brief Definitions and functions about Brain Analysis.
 * @author Samuel Martins
 * @date Apr 28, 2018
 */

#ifndef IFT_BRAIN_ANALYSIS_H_
#define IFT_BRAIN_ANALYSIS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "iftBoundingBox.h"
#include "iftClassification.h"
#include "iftGeometric.h"
#include "iftImage.h"
#include "iftNumPy.h"
#include "iftSeeds.h"


/**
 * @brief Applies a Grid Sampling by using the brain asymmetries.
 * 
 * Given an asymmetry map of a brain, it is thresholded in order to get the asymmetries as binary objects.
 * Such objects are eroded that removes small ones to decrease the number of grid points on the asymmetry
 * regions. The geometric centers of the n_points_on_asymmetries resulting objects are selected as grid points.
 * 
 * The remaining points are selected on the symmetric region. For that, the functions tries to sample
 * n_points_on_symmetries = (n_samples - n_points_on_asymmetries) points equally distanced on this regions.
 * If n_points_on_symmetries < min_samples_on_symmetries, then n_points_on_symmetries = min_samples_on_symmetries.
 * 
 * The final grid points are the union of the points on the symmetric and asymmetric regions.
 * 
 * @param asym_map Asymmetry Map of a brain.
 * @param right_hemisphere_mask Mask of the Right Hemisphere.
 * @param n_samples Number of required samples/points. Note that resulting number of samples
 *                  can be less or greater than this number. 
 * @param min_samples_on_symmetries Minimum number of super samples/points on the symmetric region.
 * @return Resulting grid points.
 * 
 * @author Samuel Martins
 * @date Oct 1, 2018
 */
iftIntArray *iftGridSamplingByBrainAsymmetries(const iftImage *asym_map, const iftImage *right_hemisphere_mask,
                                               int n_samples, int min_samples_on_symmetries);


/**
 * @brief Builds a Multi-Band Image by stacking the right and the flipped left hemisphere from an input image.
 * 
 * @attention The function assumes that the image's Mid-Sagittal Plane is its central xslice.
 * 
 * The Multi-Band Image has 2 bands: the first is the input image, and the second is the flipped input image.
 * By assuming that the mid-sagittal plane is the central xslice of the input image, then each voxel
 * will have two values: one from the right hemisphere and the other from the flipped left hemisphere.
 * 
 * @param img Brain image.
 * @return Multi-Band Image with the stacked brain hemispheres.
 * 
 * @author Samuel Martins
 * @date Sep 1, 2018
 */
iftMImage *iftBuildBrainHemisphereMImage(const iftImage *img);


/**
 * @brief Finds corresponding regions between the brain hemispheres for brain asymmetry analysis
 * by extracting supervoxels inside them.
 * 
 * @attention The function assumes that the image's Mid-Sagittal Plane is its central xslice.
 * 
 * To find/determine corresponding regions in both hemispheres for brain asymmetry analysis, it firstly
 * builds a multi-band image with 2 bands by stacking the right hemisphere with the flipped left hemisphere,
 * so that both hemispheres have the same orientation.
 * Such MImage has voxels with 2 values, one for each hemisphere.
 * 
 * Then, it extracts supervoxels from this MImage by using ISF only inside the right hemisphere, which is defined
 * by the mask <right_hemisphere_mask>.
 * The resulting supervoxels/regions that associates both hemispheres are mirrored/flipped so that
 * the resulting label image has the same supervoxels in both hemispheres.
 * 
 * The ISF's seed initialization uses the hybrid grid sampling (@see iftHybridGridSamplingOnMask) with
 * the image of absolute difference between the hemispheres. If a bias is passed, it is removed from
 * the the absolute difference (absdiff) of the hemispheres. If its standard deviation is passed, it is also
 * removed from the absdiff.
 * 
 * @param img Input Brain Image.
 * @param right_hemisphere_mask Mask that defines the right hemisphere.
 * @param n_supervoxels Required number of supervoxels.
 * @param alpha Alpha factor of ISF.
 * @param beta Beta factor of ISF.
 * @param bias (Optional) If != NULL, this bias is removed from the absolute difference of the hemispheres.
 * @param bias_stdev (Optional) If != NULL, this standard deviation of the bias is removed from the absolute difference of the hemispheres.
 * @return Label image with the resulting supervoxels.
 * 
 * @author Samuel Martins
 * @date Sep 1, 2018
 */
iftImage *iftExtractBrainSupervoxelsByISF_ROOT(const iftImage *img, const iftImage *right_hemisphere_mask,
                                               int n_supervoxels, float alpha, float beta,
                                               const iftImage *bias, const iftImage *bias_stdev);


/**
 * @brief Computes the absolute difference between the brain hemispheres from an image.
 * 
 * @attention The function assumes that the image's Mid-Sagittal Plane is its central xslice.
 * 
 * @author Samuel Martins
 * @date Sep 17, 2018
 */
inline iftImage *iftBrainHemisphereAbsDiff(const iftImage *img) {
    iftImage *img_flip = iftFlipImage(img, IFT_AXIS_X);
    iftImage *absdiff = iftAbsSub(img, img_flip);
    iftDestroyImage(&img_flip);
    return absdiff;
}

/**
 * @brief Computes the square difference between the brain hemispheres from an image.
 * 
 * @attention The function assumes that the image's Mid-Sagittal Plane is its central xslice.
 * 
 * @author Samuel Martins
 * @date Sep 17, 2018
 */
inline iftImage *iftBrainHemisphereSquareDiff(const iftImage *img) {
    iftImage *img_flip = iftFlipImage(img, IFT_AXIS_X);
    iftImage *square_diff = iftSquareSub(img, img_flip);
    iftDestroyImage(&img_flip);
    return square_diff;
}


/**
 * @brief Computes the Mean Absolute Error (MAE) for brain asymmetries from an image set.
 * 
 * For each image from the image set <img_set>, it computes the absolute difference of the voxels
 * between the right hemisphere and flipped left hemisphere. It then returns the mean image of all these differences.
 * 
 * @attention The function assumes that all the images have the same domain and orientation, and their Mid-Sagittal Plane
 * is the central xslice of each one.
 * 
 * @param img_set Set of images.
 * @return Mean Absolute Error for Brain Asymmetries.
 * 
 * @author Samuel Martins
 * @date Sep 17, 2018
 */
iftImage *iftBrainAsymmetryMAE(const iftFileSet *img_set);


/**
 * @brief Computes the Mean Absolute Error (MAE) and its standard deviation for brain asymmetries from an image set.
 * 
 * For each image from the image set <img_set>, it computes the absolute difference of the voxels 
 * between the right hemisphere and flipped left hemisphere, and its standard deviation too.
 * It then returns the mean image of all these differences.
 * 
 * @attention The function assumes that all the images have the same domain and orientation, and their Mid-Sagittal Plane
 * is the central xslice of each one.
 * 
 * @param img_set Set of images.
 * @return Mean Absolute Error for Brain Asymmetries.
 * 
 * @author Samuel Martins
 * @date Sep 17, 2018
 */
iftImage *iftBrainAsymmetryMAEStdev(const iftFileSet *img_set, iftImage **stdev_out);


iftImage *iftBrainTemplateAsymmetryMAE(const iftFileSet *img_set, const iftImage *template_img);


/**
 * @brief Computes the brain asymmetry map for the image <img>.
 * 
 * The map consists of:
 * (1) Absolute Difference between the hemispheres
 * (2) Subtraction of the bias (mean absolute error for brain asymmetries from a training set @see iftComputeBrainAsymmetryMAE)
 * (3) ReLU
 * 
 * @attention The function assumes that the image and the bias have the same domain and orientation, and their Mid-Sagittal Plane
 * is the central xslice of each one.
 * 
 * @param img Image for asymmetry map computation.
 * @param bias Bias previously computed from the mean absolute error for brain asymmetries from a training set.
 * @return Asymmetry Map.
 * 
 * @author Samuel Martins
 * @date Sep 17, 2018
 */
iftImage *iftBrainAsymmetryMap(const iftImage *img, const iftImage *bias, const iftImage *bias_stdev);

iftImage *iftBrainAsymmetryMapWithTemplate(const iftImage *img, const iftImage *template_img,
                                           const iftImage *bias_template);


/**
 * @brief Computes the medians of the asymmetries in a image set.
 * 
 * @attention The function assumes that the image and the bias have the same domain and orientation, and their Mid-Sagittal Plane
 * is the central xslice of each one.
 * 
 * @param img_set Set of images.
 * @return Image with the medians of the asymmetries.
 * 
 * @author Samuel Martins
 * @date Sep 17, 2018
 */
iftImage *iftBrainMedianAsymmetry(const iftFileSet *img_set);

#ifdef __cplusplus
}
#endif

#endif










