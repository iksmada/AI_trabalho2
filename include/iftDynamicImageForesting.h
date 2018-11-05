//
// Created by jookuma on 24/01/18.
//

#ifndef IFT_IFTDYNAMICIMAGEFORESTING_H
#define IFT_IFTDYNAMICIMAGEFORESTING_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftImage.h"
#include "iftMImage.h"
#include "iftGQueue.h"
#include "iftFHeap.h"
#include "iftMemory.h"
#include "iftLabeledSet.h"
#include "iftDataSet.h"
#include "iftKmeans.h"
#include "iftClustering.h"

#define DEC_CORRECTION 10000.0f

/**
 * @brief Structure for Dynamic Image Foresting Transform
 * @author Jordão Bragantini
 */

typedef struct ift_dynamic_tree {
    int* tree;              // array of indexes of nodes
    int size;               // current size of tree
    int n;                  // maximum number of nodes
    int* mean;              // mean of tree for each band
    int m;                  // number of bands
    int tree_label;         // label of this tree, -1 if there is no label;
    iftFHeap* heap;         // pointer to heap for tree updating and sorting
    float* dyn_path_val; // pointer to dynamic path value map
    int* aux_array; // auxiliary array for tree sorting
} iftDynamicTree;

/**
 * @brief Initializes Dynamic Tree
 * @author Jordão Bragantini
 *
 * @param  n_nodes          Maximum number of nodes
 * @param  m_bands          Number of Multi-band image bands
 * @param  tree_label       Label of this tree
 * @param  heap_queue       Heap Queue to be used when updating Tree
 * @param  dyn_path_val     Path-value map to be used when updating Tree
 * @param  aux_array        Auxiliary array to be used when updating Tree
 * @return                  Dynamic Tree Object
 */

iftDynamicTree *iftCreateDynamicTree(int n_nodes, int m_bands, int tree_label, iftFHeap *heap_queue, iftFImage* dyn_path_val, int *aux_array);

/**
 * @brief Destroys Dynamic Tree
 * @author Jordão Bragantini
 *
 * @param  T                Address of Dynamic Tree Pointer
 */
void iftDestroyDynamicTree(iftDynamicTree** T);

/**
 * @brief Insert node to Dynamic Tree T and update the mean of T and update it
 * @author Jordão Bragantini
 *
 * @param mimg              Multi-band image
 * @param node              Node to be added
 * @param  T                Address of Dynamic Tree Pointer
 * @return                  false if T is full, true otherwise
 */

bool iftInsertNodeDynamicTree(iftMImage* mimg, int node, iftDynamicTree* T);

/**
 * @brief Private function, updates and sorts T
 * @author Jordão Bragantini
 *
 * @param mimg              Multi-band image
 * @param  T                Address of Dynamic Tree Pointer
 */
void _iftSortedUpdateDynamicTree(iftMImage *mimg, iftDynamicTree* T);

/**
 * @brief Private function, return arc-weight of q
 * @author Jordão Bragantini
 *
 * @param mimg              Multi-band image
 * @param  T                Address of Dynamic Tree Pointer
 * @param q                 Node which the arc-weight will be computed
 * @return                  Arc-weight of q
 */
float _iftArcWeightByDynamicTree(iftMImage *mimg, iftDynamicTree* T, int q);

/**
 * @brief Computes image segmentation using predecessors as arcweight estimation
 * @author Jordão Bragantini
 *
 * @param  mimg             Multi-band image.
 * @param  seeds            Seeds for image segmentation.
 * @param  max_array        Max quantity of predecessor to use for arcweight estimation
 * @return                  Segmentated Image
 */

//! swig(newobject)
iftImage *iftDynamicObjectDelineation(iftMImage *mimg, iftLabeledSet *seeds, const iftAdjRel *A, int max_pred_length);

//! swig(newobject)
iftImage *iftDynamicArcWeightsForest(iftMImage *mimg, iftLabeledSet *seeds, const iftAdjRel *A,
                                     int max_pred_length, float k_perc);
//! swig(newobject)
iftImage *iftDynamicObjectDelinForest(iftMImage *mimg, iftLabeledSet *seeds, const iftAdjRel *A,
                                      int max_pred_length, float k_perc);

void iftControlClusterPurity(iftDataSet* Z, iftDynamicTree** forest, iftImage* control, float k_perc);
iftDataSet*_iftCreateForestRootsDataSet(iftLabeledSet* seeds, iftDynamicTree** forest);
void _iftUpdateDataSetFromForest(iftDataSet* Z, iftDynamicTree** forest, iftLabeledSet* seeds);

#ifdef __cplusplus
}
#endif

#endif //IFT_IFTDYNAMICIMAGEFORESTING_H
