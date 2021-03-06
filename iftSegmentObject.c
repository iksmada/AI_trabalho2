#include "ift.h"
#include <getopt.h>

#define INITIAL_LABEL 2

/* Draw seeds on the image */

void iftMyDrawBinaryLabeledSeeds(iftImage *img,iftLabeledSet *seeds,iftColor YCbCr,iftAdjRel *A)
{
    iftColor YCbCr_compl;
    int Imax = iftNormalizationValue(iftMaximumValue(img));

    YCbCr_compl.val[0] = YCbCr.val[0];
    YCbCr_compl.val[1] = Imax-YCbCr.val[1];
    YCbCr_compl.val[2] = Imax-YCbCr.val[2];

    iftLabeledSet *S = seeds;
    while (S != NULL) {
        int p = S->elem;
        iftVoxel u = iftGetVoxelCoord(img,p);
        if (S->label == 1){
            iftDrawPoint(img,u,YCbCr,A,Imax);
        } else {
            iftDrawPoint(img,u,YCbCr_compl,A,Imax);
        }
        S = S->next;
    }
}

/* Compute the maximum arc weight of the image graph */

float iftMaxArcWeight(iftMImage *mimg, iftAdjRel *A)
{
    float      Featp[mimg->m], Featq[mimg->m], maxarcw=IFT_INFINITY_FLT_NEG;


    for (int p=0; p < mimg->n; p++){
        iftVoxel u = iftMGetVoxelCoord(mimg,p);
        for (int i=1; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A,u,i);
            if (iftMValidVoxel(mimg,v)){
                int q = iftMGetVoxelIndex(mimg,v);
                for (int f=0; f < mimg->m; f++) {
                    Featp[f] = mimg->band[f].val[p];
                    Featq[f] = mimg->band[f].val[q];
                }
                float fdist = iftFeatDistance(Featp,Featq,mimg->m);
                if (fdist > maxarcw)
                    maxarcw = fdist;
            }
        }
    }

    return(maxarcw);
}

/* Compute gradient of the image graph */

iftFImage *iftGradientImage(iftMImage *mimg, iftAdjRel *A) {
    float Featp[mimg->m], Featq[mimg->m], fmax, fdist;
    iftFImage *gradient = iftCreateFImage(mimg->xsize, mimg->ysize, mimg->zsize);

    for (int p = 0; p < mimg->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mimg, p);
        fmax = 0.0;
        for (int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mimg, v)) {
                int q = iftMGetVoxelIndex(mimg, v);
                for (int f = 0; f < mimg->m; f++) {
                    Featp[f] = mimg->band[f].val[p];
                    Featq[f] = mimg->band[f].val[q];
                }
                fdist = iftFeatDistance(Featp, Featq, mimg->m);
                if (fdist > fmax)
                    fmax = fdist;
            }
        }
        gradient->val[p] = fmax;
    }

    return (gradient);
}

/* Compute a weight image from the arc weights of the image graph */

iftFImage *iftArcWeightImage(iftFImage *gradient, iftImage *objmap, float alpha, iftAdjRel *A)
{
    float fmax, fdist;
    iftFImage *weight = iftCreateFImage(objmap->xsize,objmap->ysize,objmap->zsize);

    float Wmax = iftFMaximumValue(gradient);
    float Omax = iftMaximumValue(objmap);

    for (int p=0; p < objmap->n; p++){
        iftVoxel u = iftGetVoxelCoord(objmap,p);
        fmax = 0.0;
        for (int i=1; i < A->n; i++){
            iftVoxel v = iftGetAdjacentVoxel(A,u,i);
            if (iftValidVoxel(objmap,v)){
                int q = iftGetVoxelIndex(objmap,v);
                fdist = fabs(objmap->val[q]-objmap->val[p]);
                if (fdist > fmax)
                    fmax = fdist;
            }
        }
        weight->val[p] = Omax*((gradient->val[p]/Wmax)*(1.0-alpha)+
                               alpha*fmax/Omax);
    }

    return(weight);
}

/* Computes the object map */

iftImage *iftObjectMap(iftMImage *mimg, iftLabeledSet *training_set, int Imax)
{
    iftImage *objmap=NULL;

    if (iftNumberOfLabels(training_set) != 2)
        iftError("It only works for binary segmentation","iftObjectMap");

    iftDataSet *Z1 = iftMImageSeedsToDataSet(mimg, training_set);
    iftSetStatus(Z1,IFT_TRAIN);

    iftCplGraph *graph   = iftCreateCplGraph(Z1);
    iftSupTrain(graph);

    iftDataSet *Z   = iftMImageToDataSet(mimg, NULL);
    iftSetStatus(Z,IFT_TEST);
    iftClassifyWithCertaintyValues(graph, Z);
    iftImage  *aux  = iftDataSetObjectMap(Z, NULL, Imax, 2);

    iftDestroyDataSet(&Z1);
    iftDestroyDataSet(&Z);
    iftDestroyCplGraph(&graph);

    /* post-processing */

    iftAdjRel *A=NULL;
    if (iftIs3DMImage(mimg))
        A = iftSpheric(sqrtf(3.0));
    else
        A = iftCircular(sqrtf(2.0));

    objmap = iftMedianFilter(aux,A);
    iftDestroyImage(&aux);
    iftDestroyAdjRel(&A);
    iftDestroyCplGraph(&graph);

    return(objmap);
}

/* This function must compute a new seed set, which includes the
   previous set and the pixels in the optimum paths from one arbitrary
   seed p0 to all others according to the following connectivity
   function: f(<p0>) = 0, f(<p>) = infinity for p different from p0,
   and f(path_p . <p,q>) = Omax - O(q) where O(q) is the object map
   value of q and Omax is the maximum value in the object map O. */


iftLabeledSet *iftConnectInternalSeeds(iftLabeledSet *seeds, iftImage *objmap)
{
    iftImage   *pathval = NULL, *pred = NULL;
    iftGQueue  *Q = NULL;
    int         i, p, q, tmp, Omax=iftMaximumValue(objmap);
    iftVoxel    u, v;
    iftLabeledSet *S = NULL, *newS=NULL;
    iftAdjRel     *A = NULL;

    if (iftNumberOfLabels(seeds)!=2)
        iftError("It is only implemented for binary segmentation","iftConnectInternalSeeds");

    if (iftIs3DImage(objmap))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);

    // Initialization
    pathval  = iftCreateImage(objmap->xsize, objmap->ysize, objmap->zsize);
    pred     = iftCreateImage(objmap->xsize, objmap->ysize, objmap->zsize);
    Q        = iftCreateGQueue(Omax+1, objmap->n, pathval->val);

    for (p = 0; p < objmap->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
    }

    S = seeds;
    while (S != NULL)
    {
        p              = S->elem;
        iftInsertLabeledSet(&newS,p,S->label);
        S              = S->next;
    }

    S = seeds;
    while (S != NULL)
    {
        p = S->elem;
        if (S->label > 0){
            pred->val[p]    = IFT_NIL;
            pathval->val[p] = 0;
            iftInsertGQueue(&Q,p);
            break;
        }
        S = S->next;
    }

    /* Image Foresting Transform */

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(objmap, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(objmap, v))
            {
                q = iftGetVoxelIndex(objmap, v);
                if (Q->L.elem[q].color != IFT_BLACK)
                {
                    tmp = Omax - objmap->val[q];
                    if (tmp < pathval->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q,q);
                        pred->val[q]     = p;
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&pathval);

    S = seeds;
    while (S != NULL){
        p = S->elem;
        if (S->label > 0){
            q = p;
            while (pred->val[q] != IFT_NIL){
                if(iftLabeledSetHasElement(newS, q)==0) {
                    iftInsertLabeledSet(&newS,q,1);
                }
                q = pred->val[q];
            }
        }
        S = S->next;
    }

    iftDestroyImage(&pred);

    return (newS);
}

iftImage *iftDelineateObjectByOrientedWatershed(iftImage *gradient, iftImage *objmap, iftLabeledSet *seeds) {
    iftWarning("Using this", "iftDelineateObjectByOrientedWatershed");

    iftImage   *pathval = NULL, *label = NULL;
    iftGQueue  *Q = NULL;
    int         i, p, q, tmp, Omax=iftMaximumValue(gradient);
    iftVoxel    u, v;
    iftLabeledSet *S = NULL;
    iftAdjRel     *A = NULL;
    float factor;

    if (iftNumberOfLabels(seeds)!=2)
        iftError("It is only implemented for binary segmentation","iftDelineateObjectRegion");

    if (iftIs3DImage(gradient))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);

    // Initialization
    pathval  = iftCreateImage(gradient->xsize, gradient->ysize, gradient->zsize);
    label    = iftCreateImage(gradient->xsize, gradient->ysize, gradient->zsize);
    Q        = iftCreateGQueue(Omax+1, gradient->n, pathval->val);

    for (p = 0; p < gradient->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
        //invalid label to show if some pixel is not labeled
        label->val[p] = INITIAL_LABEL;
    }

    S = seeds;
    while (S != NULL)
    {
        p = S->elem;
        label->val[p]    = S->label;
        pathval->val[p] = 0;
        iftInsertGQueue(&Q,p);
        S = S->next;
    }

    /* Image Foresting Transform */

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(gradient, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(gradient, v))
            {
                q = iftGetVoxelIndex(gradient, v);
                if (Q->L.elem[q].color != IFT_BLACK)
                {
                    //1 == Si (internal seed)
                    //0 == Se (external seed)
                    if (
                            (objmap->val[p]>objmap->val[q] && label->val[p] == 1) ||
                            (objmap->val[p]<objmap->val[q] && label->val[p] == 0)
                    )
                        factor = 1.5;
                    else
                        factor = 1.0;

                    tmp = iftMax(
                            iftRound(pow(abs(gradient->val[q]-gradient->val[p]), factor)),
                            pathval->val[p]);

                    if (tmp < pathval->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q,q);
                        label->val[q]    = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&pathval);

    return (label);
}

/* This function must delineate the object from internal and external
   seeds as described in the slides of the segmentation lectures */

iftImage *iftDelineateObjectRegion(iftImage *weight, iftImage *objmap, iftLabeledSet *seeds, float alpha) {
    iftWarning("Using this", "iftDelineateObjectRegion");

    iftImage   *pathval = NULL, *label = NULL;
    iftGQueue  *Q = NULL;
    int         i, p, q, tmp, Omax=iftMaximumValue(weight);
    iftVoxel    u, v;
    iftLabeledSet *S = NULL;
    iftAdjRel     *A = NULL;

    if (iftNumberOfLabels(seeds)!=2)
        iftError("It is only implemented for binary segmentation","iftDelineateObjectRegion");

    if (iftIs3DImage(weight))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);

    // Initialization
    pathval  = iftCreateImage(weight->xsize, weight->ysize, weight->zsize);
    label     = iftCreateImage(weight->xsize, weight->ysize, weight->zsize);
    Q        = iftCreateGQueue(Omax+1, weight->n, pathval->val);

    for (p = 0; p < weight->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
        //invalid label to show if some pixel is not labeled
        label->val[p] = INITIAL_LABEL;
    }

    S = seeds;
    while (S != NULL)
    {
        p = S->elem;
        label->val[p]    = S->label;
        pathval->val[p] = 0;
        iftInsertGQueue(&Q,p);
        S = S->next;
    }

    /* Image Foresting Transform */

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(weight, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(weight, v))
            {
                q = iftGetVoxelIndex(weight, v);
                if (Q->L.elem[q].color != IFT_BLACK)
                {
                    tmp = iftMax(
                            iftRound(
                                    alpha*abs(objmap->val[q]-objmap->val[p]) +
                                    (1-alpha)*abs(weight->val[q]-weight->val[p])
                                    ),
                            pathval->val[p]);
                    if (tmp < pathval->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q,q);
                        label->val[q]     = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&pathval);

    return (label);
}

iftImage *iftDelineateObjectByGradient(iftImage *gradient, iftLabeledSet *seeds) {
    iftWarning("Using this", "iftDelineateObjectByGradient");

    iftImage   *pathval = NULL, *label = NULL;
    iftGQueue  *Q = NULL;
    int         i, p, q, tmp, Omax=iftMaximumValue(gradient);
    iftVoxel    u, v;
    iftLabeledSet *S = NULL;
    iftAdjRel     *A = NULL;

    if (iftNumberOfLabels(seeds)!=2)
        iftError("It is only implemented for binary segmentation","iftDelineateObjectByGradient");

    if (iftIs3DImage(gradient))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);

    // Initialization
    pathval  = iftCreateImage(gradient->xsize, gradient->ysize, gradient->zsize);
    label     = iftCreateImage(gradient->xsize, gradient->ysize, gradient->zsize);
    Q        = iftCreateGQueue(Omax+1, gradient->n, pathval->val);

    for (p = 0; p < gradient->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
        //invalid label to show if some pixel is not labeled
        label->val[p] = INITIAL_LABEL;
    }

    S = seeds;
    while (S != NULL)
    {
        p = S->elem;
        label->val[p]    = S->label;
        pathval->val[p] = 0;
        iftInsertGQueue(&Q,p);
        S = S->next;
    }

    /* Image Foresting Transform */

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(gradient, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(gradient, v))
            {
                q = iftGetVoxelIndex(gradient, v);
                if (Q->L.elem[q].color != IFT_BLACK)
                {
                    //tmp = gradient->val[q];
                    tmp = iftMax(abs(gradient->val[q]-gradient->val[p]),pathval->val[p]);
                    if (tmp < pathval->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q,q);
                        label->val[q]    = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&pathval);

    return (label);
}

iftImage *iftDelineateObjectByWatershed(iftImage *gradient, iftLabeledSet *seeds) {
    iftWarning("Using this", "iftDelineateObjectByWatershed");

    iftImage   *pathval = NULL, *label = NULL;
    iftGQueue  *Q = NULL;
    int         i, p, q, tmp, Omax=iftMaximumValue(gradient);
    iftVoxel    u, v;
    iftLabeledSet *S = NULL;
    iftAdjRel     *A = NULL;

    if (iftNumberOfLabels(seeds)!=2)
        iftError("It is only implemented for binary segmentation","iftDelineateObjectRegion");

    if (iftIs3DImage(gradient))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);

    // Initialization
    pathval  = iftCreateImage(gradient->xsize, gradient->ysize, gradient->zsize);
    label     = iftCreateImage(gradient->xsize, gradient->ysize, gradient->zsize);
    Q        = iftCreateGQueue(Omax+1, gradient->n, pathval->val);

    for (p = 0; p < gradient->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
        //invalid label to show if some pixel is not labeled
        label->val[p] = INITIAL_LABEL;
    }

    S = seeds;
    while (S != NULL)
    {
        p = S->elem;
        label->val[p]    = S->label;
        pathval->val[p] = 0;
        iftInsertGQueue(&Q,p);
        S = S->next;
    }

    /* Image Foresting Transform */

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        u = iftGetVoxelCoord(gradient, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftValidVoxel(gradient, v))
            {
                q = iftGetVoxelIndex(gradient, v);
                if (Q->L.elem[q].color != IFT_BLACK)
                {
                    //tmp = gradient->val[q];
                    tmp = iftMax(abs(gradient->val[q]),pathval->val[p]);
                    if (tmp < pathval->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q,q);
                        label->val[q]    = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&pathval);

    return (label);
}

iftImage *iftDelineateObjectByDynamicArcWeight(iftMImage *mimg, iftLabeledSet *seeds, char *mode) {
    iftWarning(mode, "iftDelineateObjectByDynamicArcWeight");

    iftImage   *pathval = NULL, *label = NULL, *aux = NULL, *elements = NULL, *root = NULL;
    iftMImage  *mean = NULL;
    iftGQueue  *Q = NULL;
    int         i, p, q, tmp, s, r;
    float Featp[mimg->m], Featq[mimg->m], Feats[mimg->m], fmin, fdist, Omax = iftMMaximumValue(mimg,0);
    iftVoxel    u, v;
    iftLabeledSet *S = NULL;
    iftAdjRel     *A = NULL;
    bool w1=false,w2=false,w4=false,w5=false;

    if (iftNumberOfLabels(seeds)!=2)
        iftError("It is only implemented for binary segmentation","iftDelineateObjectByDynamicArcWeight");

    if (strcmp(mode,"w1") == 0)
        w1 = true;
    if (strcmp(mode,"w2") == 0)
        w2 = true;
    if (strcmp(mode,"w4") == 0) {
        w1 = true;
        w4 = true;
    }
    if (strcmp(mode,"w5") == 0) {
        w2 = true;
        w5 = true;
    }

    aux = iftMImageToImage(mimg,255,0);
    if (iftIs3DImage(aux))
        A = iftSpheric(1.0);
    else
        A = iftCircular(1.0);
    iftDestroyImage(&aux);

    // Initialization
    pathval   = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    label     = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    mean      = iftCreateMImage(mimg->xsize, mimg->ysize, mimg->zsize, mimg->m);
    elements  = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    root      = iftCreateImage(mimg->xsize, mimg->ysize, mimg->zsize);
    Q         = iftCreateGQueue(iftRound(Omax)+1, mimg->n, pathval->val);

    for (p = 0; p < mimg->n; p++)
    {
        pathval->val[p] = IFT_INFINITY_INT;
        //invalid label to show if some pixel is not labeled
        label->val[p] = INITIAL_LABEL;
        for (int f = 0; f < mimg->m; f++) {
            mean->band[f].val[p] = 0.0f;
        }
        elements->val[p] =  0;
        root->val[p] = -1;
    }

    S = seeds;
    while (S != NULL)
    {
        p = S->elem;
        label->val[p]    = S->label;
        pathval->val[p] = 0;
        for (int f = 0; f < mimg->m; f++) {
            mean->band[f].val[p] = mimg->band[f].val[p];
        }
        elements->val[p] = 1;
        root->val[p] = p;
        iftInsertGQueue(&Q,p);
        S = S->next;
    }

    S = seeds;

    /* Image Foresting Transform */

    while (!iftEmptyGQueue(Q))
    {
        p = iftRemoveGQueue(Q);
        //update elements and mean tables
        r = root->val[p];
        elements->val[r] = elements->val[r] + 1;
        for (int f = 0; f < mean->m; f++) {
            //calc mean dynamictly https://math.stackexchange.com/questions/106700/incremental-averageing
            mean->band[f].val[r] = mean->band[f].val[r] +
                                   (mimg->band[f].val[p] - mean->band[f].val[r])/elements->val[r];
        }

        u = iftMGetVoxelCoord(mimg, p);

        for (i = 1; i < A->n; i++)
        {
            v = iftGetAdjacentVoxel(A, u, i);

            if (iftMValidVoxel(mimg, v))
            {
                q = iftMGetVoxelIndex(mimg, v);
                if (Q->L.elem[q].color != IFT_BLACK)
                {

                    fmin = IFT_INFINITY_FLT;
                    for (int f = 0; f < mimg->m; f++) {
                        Featq[f] = mimg->band[f].val[q];
                    }
                    // calculate w2 min{||Ur - Iq||}
                    if (w2) {
                        while (S != NULL) {
                            s = S->elem;
                            if (S->label == label->val[p]) {
                                for (int f = 0; f < mean->m; f++) {
                                    Feats[f] = mean->band[f].val[s];
                                }
                                fdist = iftFeatDistance(Feats, Featq, mean->m);
                                fmin = iftMin(fdist, fmin);
                            }
                            S = S->next;
                        }
                        S = seeds;
                    }
                    if (w1) {
                        for (int f = 0; f < mean->m; f++) {
                            Feats[f] = mean->band[f].val[r];
                        }
                        fmin = iftFeatDistance(Feats, Featq, mean->m);
                    }

                    if (w4 || w5) {
                        // calculate w5 ||Iq - Ip||
                        for (int f = 0; f < mimg->m; f++) {
                            Featp[f] = mimg->band[f].val[p];
                        }
                        fdist = iftFeatDistance(Featp, Featq, mimg->m);
                    } else
                        fdist=0.0f;

                    //compare
                    tmp = iftRound(iftMax(fmin + fdist, pathval->val[p]));
                    if (tmp < pathval->val[q]){
                        if (Q->L.elem[q].color == IFT_GRAY)
                            iftRemoveGQueueElem(Q,q);
                        label->val[q]    = label->val[p];
                        pathval->val[q]  = tmp;
                        iftInsertGQueue(&Q, q);
                        //update root
                        root->val[q] = r;

                    }
                }
            }
        }
    }

    iftDestroyAdjRel(&A);
    iftDestroyGQueue(&Q);
    iftDestroyImage(&pathval);
    iftDestroyMImage(&mean);
    iftDestroyImage(&elements);
    iftDestroyImage(&root);

    return (label);
}

int main(int argc, char *argv[]) {
    iftAdjRel *A = iftCircular(1.0);
    iftAdjRel *B = iftCircular(0.0);
    iftAdjRel *C = iftCircular(sqrtf(2.0));
    iftColor RGB, Blue, Red, Green;
    float alpha;
    static int region, watershed, oriented_watershed, dynamic_arc_weight, gradient_flag, connect_seeds, draw_seeds;
    char *alpha_str, *input, *output, *seeds_path, *mode = iftAllocString(2);

    if (argc < 5) {
        iftError("Usage: iftSegmentObject <input-image.png> <training-set.txt> <alpha [0-1]> <output-label.png>",
                 "main");
        return (1);
    } else {
        input = argv[1];
        seeds_path = argv[2];
        alpha_str = argv[3];
        output = argv[4];
        if (argc > 5) {
            static struct option long_options[] =
                    {
                            {"region", no_argument, &region,             1},
                            {"watershed", no_argument, &watershed,          1},
                            {"oriented-watershed", no_argument, &oriented_watershed, 1},
                            {"dynamic-arc-weight", required_argument, 0, 'd'},
                            {"gradient", no_argument, &gradient_flag, 1},
                            {"connect-seeds", no_argument, &connect_seeds, 1},
                            {"draw-seeds", no_argument, &draw_seeds, 1},
                    };
            /* getopt_long stores the option index here. */
            int option_index = 4, opt;

            while ((opt = getopt_long_only(argc, argv, "d:", long_options, &option_index)) != -1) {
                switch (opt) {
                    case 'd':
                        strcpy(mode,optarg);
                        dynamic_arc_weight=true;
                    default:
                        break;
                }
            }
        }
    }

    alpha = atof(alpha_str);
    if ((alpha < 0.0) || (alpha > 1.0))
        iftError("alpha=%f is outside [0,1]", "main", alpha);


    /* Read image and pre-process it to reduce noise */
    iftImage *aux = iftReadImageByExt(input);
    iftImage *img = iftSmoothImage(aux, C, 3.0);
    iftDestroyImage(&aux);

    /* Compute normalization value to combine weights and visualize overlays */

    int Imax = iftNormalizationValue(iftMaximumValue(img));
    RGB.val[0] = Imax / 5.0;
    RGB.val[1] = Imax / 2.0;
    RGB.val[2] = Imax;
    Blue = iftRGBtoYCbCr(RGB, Imax);

    RGB.val[0] = Imax;
    RGB.val[1] = Imax / 5.0;
    RGB.val[2] = Imax / 5.0;
    Red = iftRGBtoYCbCr(RGB, Imax);

    RGB.val[0] = Imax / 5.0;
    RGB.val[1] = Imax;
    RGB.val[2] = Imax / 5.0;
    Green = iftRGBtoYCbCr(RGB, Imax);

    /* Convert image into a multiband image */

    iftMImage *mimg = NULL;

    if (!iftIsColorImage(img)) {
        mimg = iftImageToMImage(img, GRAY_CSPACE);
    } else {
        mimg = iftImageToMImage(img, YCbCr_CSPACE);
    }

    /* Read seeds as training set */

    iftLabeledSet *training_set = iftReadSeeds(img, seeds_path);

    iftImage *objmap = NULL;
    if (connect_seeds || region || oriented_watershed) {
        /* Create the object map by pixel classification */
        objmap = iftObjectMap(mimg, training_set, Imax);
        iftWriteImageByExt(objmap, "objmap.png");
    }

    iftFImage *gradient = NULL;
    if (region || watershed || oriented_watershed || gradient_flag) {
        gradient = iftGradientImage(mimg, C);
        aux = iftFImageToImage(gradient, Imax);
        iftWriteImageByExt(aux, "gradient.png");
        iftDestroyImage(&aux);
    }

    iftLabeledSet *seeds = NULL;
    if (connect_seeds) {
        printf("Seeds Connected\n");
        seeds = iftConnectInternalSeeds(training_set, objmap);
        iftDestroyLabeledSet(&training_set);
    }
    else
       seeds = training_set;


    /* to exchange across the three methods, change the comments
       below. You must also add the algorithm of the dynamic IFT using
       w5 as in the paper. */

    iftImage *label = NULL;
    if (region || watershed || oriented_watershed || gradient_flag) {
        aux = iftFImageToImage(gradient, Imax);
        if (region)
            label = iftDelineateObjectRegion(aux, objmap, seeds, alpha);
        else if (watershed)
            label = iftDelineateObjectByWatershed(aux, seeds);
        else if (gradient_flag)
            label = iftDelineateObjectByGradient(aux,seeds);
        else
            label = iftDelineateObjectByOrientedWatershed(aux,objmap,seeds);
    } else
        label = iftDelineateObjectByDynamicArcWeight(mimg,seeds, mode);
    iftDestroyImage(&aux);

    /* Draw segmentation border */

    //iftDrawBorders(img, label, A, Blue, B);
    aux = iftMask(img,label);
    if (draw_seeds) {
        printf("Drawing Seeds\n");
        iftMyDrawBinaryLabeledSeeds(aux, seeds, Red, A);
    }

    iftWriteImageByExt(aux,output);

    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
    iftDestroyAdjRel(&C);
    iftDestroyImage(&img);
    iftDestroyImage(&aux);
    iftDestroyImage(&objmap);
    iftDestroyFImage(&gradient);
    iftDestroyImage(&label);
    iftDestroyMImage(&mimg);
    iftDestroyLabeledSet(&seeds);

    return(0);
}
