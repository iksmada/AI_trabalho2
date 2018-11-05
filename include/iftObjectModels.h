/**
 * @file iftObjectModels.h
 * @brief Definitions and functions about Object Models.
 * @author Samuel Martins
 * @date Jan 01, 2017
 * @ingroup ObjectModels
 */

#ifndef IFT_OBJECTMODELS_H_
#define IFT_OBJECTMODELS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "iftCommon.h"
#include "iftDict.h"
#include "iftImage.h"
#include "iftParamOptimizationProblems.h"
#include "iftFImage.h"
#include "iftNumPy.h"
#include "iftRadiometric.h"
#include "iftRegistration.h"
#include "iftSegmentation.h"
#include "iftSimilarity.h"


/**
 * @brief SOSM-S Model of a single Object.
 * @author Samuka
 * @date Dec 13, 2016
 * @ingroup ObjectModel
 */
typedef struct ift_sosms_obj_model {
    /** Label of the target Object */
    int label;
    /** (Cropped) Prior Probability Atlas */
    iftFImage *prob_atlas;
    /** Shape (domain) of the Template Image, where the object model is built */
    iftImageDomain template_shape;
    /** Begin voxel of the cropped prob atlas on template image coordinate space */
    iftVoxel begin;
    // search region for the object localization from the reference voxel
    iftBoundingBox search_region;
} iftSOSMSObjModel;


/**
 * @brief SOSM-S Model.
 * @author Samuka
 * @date Dec 13, 2016
 * @ingroup ObjectModel
 */
typedef struct ift_sosm_s {
    /** Array of labels of each SOSM-S Object Model */
    iftIntArray *labels;
    /** Array of Object Models. */
    iftSOSMSObjModel **obj_models;
    /** Template (Reference Image) where the object models are trained (if required). */
    iftImage *template_img;
} iftSOSMS;



/**
 * @brief Object Model of Adaptive Probabilistic Atlas AdaPro.
 * @author Samuka
 * @date Dec 17, 2018
 * @ingroup ObjectModel
 */
typedef struct ift_obj_model {
    /** Label of the target Object */
    int label;
    /** (Cropped) Prior Probability Atlas */
    iftFImage *prob_atlas;
    /** Shape (domain) of the Template Image, where the object model is built */
    iftImageDomain template_shape;
    /** Begin voxel of the cropped prob atlas on template image coordinate space */
    iftVoxel begin;
    /** Radius used to erode the prob. atlas to estimate the inner seeds */
    float erosion_radius;
    /** Radius used to dilate the prob. atlas to estimate the outer seeds */
    float dilation_radius;
    /** Penalizing Factor used for the delineation of the object. */
    float penalizing_factor;
} iftObjModel;


/**
 * @brief Adaptive Probabilist Atlas.
 * @author Samuka Martins
 * @date Mar 13, 2018
 */
typedef struct ift_adapro {
    /** Array of labels of each Object Model */
    iftIntArray *labels;
    /** Array of Object Models. */
    iftObjModel **obj_models;
    /** Rough Brain Mask */
    iftImage *brain_mask;
    /** Template (Reference Image) where the object models are trained (if required). */
    iftImage *template_img;
    /** Image with the training voxels (labeled voxels) for the linear SVM training. */
    iftImage *train_voxels_img;
    /** Linear SVM for texture Classification. */
    iftSVM *svm;
    /** Penalizing Factor used for the delineation of the Background. */
    float bg_penalizing_factor;
} iftAdaPro;








///////////////////////////////////// FUNCTIONS //////////////////////////////////////////////////
/**
 * @brief Put the (cropped) prob atlas of an object model on its template image's domain from its begining voxel.
 *
 * @author Samuka
 * @date Nov 16, 2017
 * @ingroup ObjectModels
 */
iftFImage *iftProbAtlasOnTemplateImageDomain(const iftFImage *prob_atlas, iftImageDomain template_shape,
                                             iftVoxel begin);


/**
 * @brief Apply the N4 algorithm for Inhomogeneity Correct in a MRI Image.
 *
 * It is used the default parameters, suggested by 3D Slicer tool [2], for N4 [1].\n
 * [1] Tustison, Nicholas J., et al. \"N4ITK: improved N3 bias correction.\" IEEE transactions on medical imaging 29.6 (2010): 1310-1320.\n
 * 
 * @param  img           Image to be corrected.
 * @param  mask          Binary mask that defines the structure of your interest. If NULL, N4 considers the entire image.
 * @param  shrink_factor Image Resampling/Shrinking Factor (1, 2, 3, 4, ...) to decrease the computation time.
 *                       Shrink factors <= 4 are commonly used.
 * @param  out_bias      The resulting Bias Field Image by the N4 correction. If NULL, it is not considered. 
 * @return               Corrected image by N4.
 *
 * @author Samuka Martins
 * @date Mar 13, 2017
 */
iftImage *iftN4BiasFieldCorrection(const iftImage *img, const iftImage *mask, int shrink_factor,
                                   iftImage **out_bias);


/**
 * @brief A good scheme to apply the N4 over the input image in a more effective way.
 *
 * Firtly, N4 is applied with shrink factor 4 and the default arguments.
 * Then, another N4 is applied with shrink factor 6 and the default arguments, in order to make
 * a fine tunning in the correction.
 * 
 * @param  img      Image to be corrected.
 * @param  mask     Binary mask that defines the structure of your interest. If NULL, N4 considers the entire image.
 * @param  out_bias The resulting Bias Field Image by the N4 correction. If NULL, it is not considered.
 * @return          Corrected image by N4.
 *
 * @author Samuka Martins
 * @date Mar 13, 2017
 */
iftImage *iftEffectiveN4BiasFieldCorrection(const iftImage *img, const iftImage *mask, iftImage **out_bias);


/**
 * @brief Apply a set of Pre-Processing operation on to a Brain MRI Image.
 *
 * The operations correspond to: 
 * (1) N4 Bias Field Correction
 * (2) Median Filter for noise reduction
 * (3) Normalization to range [0, 4095] by saturating to 4095 the 2% of brighter voxels
 * 
 * @param  mri                 Brain MR-Image
 * @param  apply_n4            Apply the N4 Bias Field Correction.
 * @param  apply_smoothing     Apply Median Filter.
 * @param  apply_normalization Apply Normalization
 * @return                     Pre-processed MR Brain Image.
 *
 * @author Samuka
 * @date Dec 20, 2017
 */
iftImage *iftMRIBrainPreProcessing(const iftImage *mri, bool apply_n4, bool apply_smoothing,
                                   bool apply_normalization);



///////////////// SOSM-S
/**
 * @brief Destroys a SOSM-S Model.
 * @author Samuka
 * @date Dec 20, 2016
 * @ingroup ObjectModels
 */
void iftDestroySOSMS(iftSOSMS **sosm_s);


/**
 * @brief Stores on disk a SOSM-S Model.
 * @author Samuka
 * @date Dec 20, 2016
 * @ingroup ObjectModels
 */
void iftWriteSOSMS(const iftSOSMS *sosm_s, const char *path);


/**
 * @brief Reads from disk a SOSM-S Model.
 * @author Samuka
 * @date Dec 20, 2016
 * @ingroup ObjectModels
 */
iftSOSMS *iftReadSOSMS(const char *path);


/**
 * @brief Trains a SOSM-S Model [1] from a set of registered segmentation masks.
 *
 * [1] Phellan, 2016 - Medical physics - Medical image segmentation via atlases and fuzzy object models
 * 
 * @param  template_img Template Image (Reference Image).
 * @param  labels       Array with the object labels
 * @return              SOSM-S Model.
 *
 * @author sAMUKA
 * @date Dec 20, 2016
 * @ingroup ObjectModels
 *
 */
iftSOSMS *iftTrainSOSMS(const iftFileSet *atlas_set, const iftImage *template_img,
                        const iftIntArray *labels);


/**
 * @brief Segments a Testing Image using the Statistical Multi-Object Shape Model SOSM-S (Phellan, 2016).
 *
 * It applies an Object Location by MSPS translating the seed models over the test image gradient. \n
 * Phellan, 2016 - Medical physics - Medical image segmentation via atlases and fuzzy object models
 * 
 * @param  test_img Test Image to be segmented.
 * @param  sosm_s   Statistical Object Shape Model.
 * @return          Segmented Image.
 * 
 * @author Samuka
 * @date Jan 3, 2016
 * @ingroup ObjectModels
 */
iftImage *iftSegmentBySOSMS(const iftImage *img, const iftImage *grad_img_in, const iftSOSMS *sosm_s);


/**
 * @brief Finds the SOSM-S Object Model's Seeds for delineation.
 * It assumes that the test image is registered on the template image (or vice-versa), used to
 * build the object model
 *
 * Given a Model (probabilistic map), background's seeds are those with prob 0 and object's seeds
 * with prob 1. \n
 * 
 * @param  test_img           Testing Image.
 * @param  obj_model          SOSM-S Object Model.
 * @return                    Labeled seeds for delineation.
 *
 * @author Samuka
 * @date Jan 3, 2017
 * @ingroup ObjectModels
 */
iftLabeledSet *iftFindSOSMSSeeds(const iftImage *test_img, const iftSOSMSObjModel *obj_model);



///////////////// MALF
/**
 * @brief      Selects the best atlases to a test image by Normalized Mutual Information [1].
 *             Train set and test image must be already registered.
 * 
 * The higher the NMI, more similar (better) the train image is. \n
 * [1] Aljabar, 2009 - Neuroimage - Multi-atlas based segmentation of brain images: atlas selection and its effect on accuracy
 *
 * @param test_img        (Registered) Testing Image.
 * @param train_img_set   (Registered) Training Image Set.
 * @param n_atlases       Number of atlases to be selected.
 * @return     A file set with the pathnames from the selected images.
 * 
 * @author Samuka
 * @date Jan 27, 2017
 * @ingroup ObjectModels
 */
iftFileSet *iftAtlasSelectionByNMI(const iftImage *test_img, const iftFileSet *train_img_set, int n_atlases);


/**
 * @brief Segments an Image by Classical MALF. This function considers the training label set is already registered on testing image.
 *
 * Classical MALF registers all atlases (img + label img) on test image's space. \n
 * The label of each voxel from segmented image is the most frequent label (Majority Voting) \n
 * 
 * @param  test_img        Testing Image to be segmented.
 * @param  train_label_set Training label set already registered on testing image's space.
 * @return                 Segmented image.
 *
 * @author Samuka
 * @date Jan 27, 2017
 * @ingroup ObjectModels
 */
iftImage *iftSegmentByClassicalMALF(const iftImage *test_img, const iftFileSet *train_label_set);


/**
 * @brief Segments an Image by MALF using STAPLE as Label Fusion.
 * This function considers that the training label set is already registered on testing image.
 *
 * @warning The CRKIT, which has the STAPLE binary program, must be installed before.
 *
 * @param  test_img        Testing Image to be segmented.
 * @param  train_label_set Training label set already registered on testing image's space.
 * @return                 Segmented image.
 *
 * @author Samuka
 * @date Jan 27, 2017
 * @ingroup ObjectModels
 */
iftImage *iftSegmentByMALFSTAPLE(const iftImage *test_img, const iftFileSet *train_atlas_set);





///////////////// AdaPro
/**
 * @brief Destroy an AdaPro Model.
 * @author Samuka
 * @date Mar 13, 2018
 */
void iftDestroyAdaPro(iftAdaPro **adapro);


/**
 * @brief Read an AdaPro Model.
 * @author Samuka
 * @date Mar 13, 2018
 */
iftAdaPro *iftReadAdaPro(const char *format, ...);


/**
 * @brief Write an AdaPro Model.
 * @author Samuka
 * @date Mar 13, 2018
 */
void iftWriteAdaPro(const iftAdaPro *adapro, const char *path, ...);


/**
 * @brief Train an Adaptive Probabilistic Atlases (AdaPro).
 * 
 * Each object is indexed at a position i, so that each one of is parameter (erosion radius, ...)
 * is in the same position of the corresponding array.
 * If the label image with the training voxels <train_voxels_img> is NULL, the model won't be adaptive,
 * only relying on the shape constraints of the probabilistic atlases during segmentation
 * 
 * @param atlas_set        Set of Registered Atlases (Label Images) on the template <template_img>.
 * @param template_img     Template Image (Reference Image Space) where the atlases are registered.
 * @param train_voxels_img Image with the markers chosen on template for linear SVM Training. BG voxels has label 0 and objects 1.
 *                         If NULL, the model won't be adaptive, only relying on the shape constraints of the probabilistic atlases.
 * @param labels           Array with the labels of the target objects for training the AdaPro.
 * @param e_radius_arr     (Optional) Array with the erosion radius to generate the inner seeds for each object.
 *                         If NULL, nothing is stored.
 * @param d_radius_arr     (Optional) Array with the dilation radius to generate the outer seeds for each object.
 *                         If NULL, nothing is stored.
 * @param C                Parameter C for linear SVM training.
 * @return                 The trained AdaPro.
 * 
 * @author Samuka Martins
 * @date Mar 13, 2018
 */
iftAdaPro *iftTrainAdaPro(const iftFileSet *atlas_set, const iftImage *template_img, const iftImage *train_voxels_img,
                          const iftIntArray *labels, const iftFloatArray *pen_factors,
                          const iftFloatArray *e_radius_arr, const iftFloatArray *d_radius_arr,
                          double C);


/**
 * @brief Segment a test image by AdaPro. The test image and the adapro's template must be in the same
 * coordinate space.
 * @param  img                          Image to be segmented.
 * @param  adapro                       Adaptive Probabilistic Atlas.
 * @param  skip_texture_classification  If true, the a texture classification is ignored.
 *                                      Otherwise, only the shape-based segmentation is performed.
 * @return                              Segmented Image.
 *
 * @author Samuka
 * @date Mar 13, 2018
 */
iftImage *iftSegmentByAdaPro(const iftImage *img, const iftAdaPro *adapro, bool skip_texture_classification);
iftImage *iftSegmentByAdaPro_Old(const iftImage *img, const iftAdaPro *adapro, bool skip_texture_classification);



/**
 * @brief Register (by Elastix) a AdaPro Model on a test image's space. Resulting registrations
 * and mapping are assigned in the own input model.
 * 
 * Reference image is registerd with the test one. This will be the new reference image. \n
 * Then, all object models are mapped to new space using the deformation fields. \n
 * Deformation fields are not saved.
 * 
 * @param adapro       AdaPro Model.
 * @param test_img     Testing Image.
 * @param elastix_files Elastix Parameter Files.
 *
 * @author Samuka
 * @date Mar 18, 2018
 * @ingroup ObjectModels
 */
void iftRegisterAdaProOnTestImageByElastix(iftAdaPro *adapro, const iftImage *test_img, const iftFileSet *elastix_files);



/**
 * @brief Finds the Object Model's Seeds for delineation.
 * It assumes that the test image is registered on the template image (or vice-versa), used to
 * build the object model
 *
 * Given a AdaPro Model (probabilistic map), background's seeds are those with prob 0 and object's seeds
 * with prob 1, after eroding and dilating the target object. \n
 * Finally, resulting seeds are filtered by a membership map (if != NULL), which is the test image previously classified,
 * where 0 are the background and 1 is the target object. \n
 * Filtering follows the rule: Inner seeds classified as background become outer seeds.
 *
 * All voxels of centain region of background or object (probability 0.0 or 1.0) are returned in a set,
 * if certain_obj_region != NULL.
 * 
 * @param  obj_model          Object Model.
 * @param  membership_map     Membership Map (test image previsously classified) used to filtered the seeds. If NULL, no filtering.
 * @param  certain_obj_region Reference to save the set with the certain region voxels of object from the prob. atlas. If NULL, nothing is returned.
 * @param  forbidden          Reference to save the set with the certain region voxels of object and background (prob of 1 and 0).
 * @return                    Labeled seeds for delineation.
 *
 * @author samuka
 * @date Mar 18, 2018
 * @ingroup ObjectModels
 */
iftLabeledSet *iftFindObjModelSeeds(const iftObjModel *obj_model, const iftImage *membership_map,
                                    iftSet **certain_obj_region_out, iftSet **forbidden);


#ifdef __cplusplus
}
#endif

#endif

