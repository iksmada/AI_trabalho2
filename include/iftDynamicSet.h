//
// Created by jordao on 05/06/18.
//

#ifndef IFT_IFTDYNAMICSET_H
#define IFT_IFTDYNAMICSET_H

#include "iftImage.h"
#include "iftLabeledSet.h"
#include "iftMImage.h"
#include "iftDataSet.h"
#include "iftDHeap.h"
#include "iftList.h"


//typedef struct ift_dyn_tree
//{
//    iftList **set;
//    double *mean;
//} iftDynTree;
//
//
//typedef struct ift_marker_forest
//{
//    iftDynTree **tree;
//    int n_roots;
//} iftMarkerForest;
//
//
//typedef struct ift_object_forest
//{
//    iftMarkerForest **mrk_forest;
//    int n_mrks;
//} iftObjectForest;
//
//
//typedef struct ift_dynamic_set
//{
//    iftObjectForest **obj_forest;
//    int n_objs;
//    iftImage *ref;
//} iftDynamicSet;


//iftDynamicSet *iftCreateDynamicSet(iftMImage* mimg, iftLabeledSet *seeds);

typedef struct ift_dynamic_set
{
    double *mean;
    int size;

} iftDynamicSet;

inline void iftInsertDynamicSet(iftDynamicSet *S, iftMImage *mimg, int p)
{
//    S->mean[0] = (S->mean[0] * S->size + mimg->band[0].val[p]) / (S->size + 1);
//    S->mean[1] = (S->mean[1] * S->size + mimg->band[1].val[p]) / (S->size + 1);
//    S->mean[2] = (S->mean[2] * S->size + mimg->band[2].val[p]) / (S->size + 1);

    S->mean[0] += (mimg->band[0].val[p] - S->mean[0]) / (S->size + 1);
    S->mean[1] += (mimg->band[1].val[p] - S->mean[1]) / (S->size + 1);
    S->mean[2] += (mimg->band[2].val[p] - S->mean[2]) / (S->size + 1);

    ++S->size;
}

iftDynamicSet *iftCreateDynamicSet(int n_bands);

void iftDestroyDynamicSet(iftDynamicSet **S);

inline double iftDistDynamicSetMImage(iftDynamicSet *S, iftMImage *mimg, int p)
{
    double dist = (S->mean[0] - mimg->band[0].val[p]) * (S->mean[0] - mimg->band[0].val[p]) +
                  (S->mean[1] - mimg->band[1].val[p]) * (S->mean[1] - mimg->band[1].val[p]) +
                  (S->mean[2] - mimg->band[2].val[p]) * (S->mean[2] - mimg->band[2].val[p]);

    return dist;
}

//! swig(newobject)
iftImage *iftDynamicSetObjectPolicy(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, bool use_dist);

//! swig(newobject)
iftImage *iftDynamicSetRootPolicy(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, int h, bool use_dist);

//! swig(newobject)
iftImage *iftDynamicSetMinRootPolicy(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, int h, bool use_dist);

//! swig(newobject)
iftImage *iftDynamicSetWithCluster(iftMImage *mimg, iftImage *cluster, iftAdjRel *A, iftLabeledSet *seeds, int h, bool use_dist);

//! swig(newobject)
iftImage *iftDynamicSetRootEnhanced(iftMImage *mimg, iftImage *objmap, iftAdjRel *A, iftLabeledSet *seeds, int h, float alpha, bool use_dist);

//! swig(newobject)
iftImage *iftDynamicSetMinRootEnhanced(iftMImage *mimg, iftImage *objmap, iftAdjRel *A, iftLabeledSet *seeds, int h, float alpha, bool use_dist);



typedef struct ift_dynamic_set_CIARP
{
    double *mean;
    int size;
} iftDynamicSet_CIARP;

inline void iftInsertDynamicSet_CIARP(iftDynamicSet_CIARP *S, iftMImage *mimg, int p)
{
    S->mean[0] += (mimg->band[0].val[p] - S->mean[0]) / (S->size + 1);
    S->mean[1] += (mimg->band[1].val[p] - S->mean[1]) / (S->size + 1);
    S->mean[2] += (mimg->band[2].val[p] - S->mean[2]) / (S->size + 1);

    ++S->size;
}

inline double iftDistDynamicSetMImage_CIARP(iftDynamicSet_CIARP *S, iftMImage *mimg, int p)
{
    double dist = (S->mean[0] - mimg->band[0].val[p]) * (S->mean[0] - mimg->band[0].val[p]) +
                  (S->mean[1] - mimg->band[1].val[p]) * (S->mean[1] - mimg->band[1].val[p]) +
                  (S->mean[2] - mimg->band[2].val[p]) * (S->mean[2] - mimg->band[2].val[p]);

    return dist;
}

iftDynamicSet_CIARP *iftCreateDynamicSet_CIARP(void);
void iftDestroyDynamicSet_CIARP(iftDynamicSet_CIARP **S);
iftImage *iftDynamicSetObjectPolicy_CIARP(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, bool use_dist);
iftImage *iftDynamicSetRootPolicy_CIARP(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, bool use_dist);
iftImage *iftDynamicSetMinRootPolicy_CIARP(iftMImage *mimg, iftAdjRel *A, iftLabeledSet *seeds, bool use_dist);


#endif //IFT_IFTDYNAMICSET_H
