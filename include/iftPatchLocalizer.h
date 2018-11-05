/**
 * @file iftBrainPatchLocalizer.h
 * @brief Definitions and functions about Brain Patch Localizer.
 * @author Samuel Martins
 * @date Jan 15, 2018
 */



#ifndef IFT_PATCH_LOCALIZER_H
#define IFT_PATCH_LOCALIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftMemory.h"
#include "iftBoundingBox.h"
#include "iftImage.h"


typedef struct ift_patch_localizer {
    iftBoundingBox patch;
    iftBoundingBox search_region;
    iftBoundingBox template_patch;
} iftPatchLocalizer;


typedef struct ift_patch_localizer_bank {
    long n;
    iftPatchLocalizer *localizers;
    iftImage *template_img;
} iftPatchLocalizerBank;


void iftDestroyPatchLocalizerBank(iftPatchLocalizerBank **bank);

iftPatchLocalizer iftTrainPatchLocalizer(const iftBoundingBox *bbs, long n);

iftPatchLocalizerBank *iftTrainPatchLocalizerBank(iftBoundingBox **patches, const iftImage *template_img,
                                                  iftBoundingBox *template_patches, long n_objs, long n_patches,
                                                  bool patch_size_power_of_two);


iftPatchLocalizerBank *iftBuildPatchLocalizerBank(const iftImage *template_img, const iftVoxel *center_voxels,
                                                  int n_center_voxels, int patch_size, int search_region_disp);


iftBoundingBoxArray *iftFindPatchesByPatchLocalizerBank(const iftImage *img, const iftPatchLocalizerBank *bank);


iftPatchLocalizerBank *iftReadPatchLocalizerBank(const char *filename, ...);

void iftWritePatchLocalizerBank(const iftPatchLocalizerBank *bank, const char *filename, ...);







#ifdef __cplusplus
}
#endif

#endif







