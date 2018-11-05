#include "ift.h"

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

/* Compute a weight image from the arc weights of the image graph */

iftFImage *iftArcWeightImage(iftMImage *mimg, iftImage *objmap, float alpha, iftAdjRel *A)
{
  float Featp[mimg->m], Featq[mimg->m], fmax=0.0, fdist;
  iftFImage *weight = iftCreateFImage(mimg->xsize,mimg->ysize,mimg->zsize);

  if ((objmap == NULL)&&(alpha != 0.0))
    iftError("It requires an object map for alpha=%f","iftArcWeightImage",alpha);

  for (int p=0; p < mimg->n; p++){
    iftVoxel u = iftMGetVoxelCoord(mimg,p);
    fmax = 0.0;
    for (int i=1; i < A->n; i++){
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);
      if (iftMValidVoxel(mimg,v)){
	int q = iftMGetVoxelIndex(mimg,v);
	for (int f=0; f < mimg->m; f++) {
	    Featp[f] = mimg->band[f].val[p];
	    Featq[f] = mimg->band[f].val[q];
	}
	fdist = iftFeatDistance(Featp,Featq,mimg->m);
	if (fdist > fmax)
	  fmax = fdist;
      }
    }
    weight->val[p] = fmax;
  }
   
  if (objmap != NULL) {
    
    float Wmax = iftFMaximumValue(weight);
    float Omax = iftMaximumValue(objmap);
    
    for (int p=0; p < mimg->n; p++){
      iftVoxel u = iftMGetVoxelCoord(mimg,p);
      fmax = 0.0;
      for (int i=1; i < A->n; i++){
	iftVoxel v = iftGetAdjacentVoxel(A,u,i);
	if (iftMValidVoxel(mimg,v)){
	  int q = iftMGetVoxelIndex(mimg,v);
	  fdist = fabs(objmap->val[q]-objmap->val[p]);
	  if (fdist > fmax)
	    fmax = fdist;
	}
      }
      weight->val[p] = Omax*((weight->val[p]/Wmax)*(1.0-alpha)+
			     alpha*fmax/Omax);
    }
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

iftImage *iftDelineateObjectByWatershed(iftFImage *weight, iftLabeledSet *seeds) {
  
  iftImage   *label = NULL;

  return (label);
}

iftImage *iftDelineateObjectByOrientedWatershed(iftFImage *weight, iftImage *objmap, iftLabeledSet *seeds) {
  
  iftImage   *label = NULL;

  return (label);
}

/* This function must delineate the object from internal and external
   seeds as described in the slides of the segmentation lectures */

iftImage *iftDelineateObjectRegion(iftMImage *mimg, iftImage *objmap, iftLabeledSet *seeds, float alpha) {
  
  iftImage   *label = NULL;


  return (label);
}



int main(int argc, char *argv[])
{  
  iftAdjRel *A=iftCircular(1.0);
  iftAdjRel *B=iftCircular(0.0);
  iftAdjRel *C=iftCircular(sqrtf(2.0));
  iftColor   RGB, YCbCr;
  float      alpha;
  
  if (argc != 5){
    iftError("Usage: iftSegmentObject <input-image.png> <training-set.txt> <alpha [0-1]> <output-label.png>","main");
  }
  alpha = atof(argv[3]); 
  if ((alpha<0.0)||(alpha>1.0))
    iftError("alpha=%f is outside [0,1]","main",alpha);
  
  
  /* Read image and pre-process it to reduce noise */
  
  iftImage  *aux = iftReadImageByExt(argv[1]);
  iftImage  *img = iftSmoothImage(aux,C,3.0);
  iftDestroyImage(&aux);

  /* Compute normalization value to combine weights and visualize overlays */
  
  int Imax = iftNormalizationValue(iftMaximumValue(img));
  RGB.val[0] = Imax/5.0;
  RGB.val[1] = Imax/2.0;
  RGB.val[2] = Imax;
  YCbCr      = iftRGBtoYCbCr(RGB,Imax);

  /* Convert image into a multiband image */
  
  iftMImage *mimg = NULL;
  
  if (!iftIsColorImage(img)) {
    mimg = iftImageToMImage(img, GRAY_CSPACE);
  } else{
    mimg = iftImageToMImage(img, YCbCr_CSPACE);
  }

  /* Read seeds as training set */
  
  iftLabeledSet *training_set = iftReadSeeds(img, argv[2]);

  /* Create the object map by pixel classification */
  
  iftImage *objmap=NULL;
  objmap = iftObjectMap(mimg, training_set, Imax);
  iftWriteImageByExt(objmap,"objmap.png");

  iftFImage *weight = iftArcWeightImage(mimg,objmap,alpha,C);
  aux  = iftFImageToImage(weight,Imax);
  iftWriteImageByExt(aux,"weight.png");
  

  /* to use or not this function, change comments below */
  iftLabeledSet *seeds = iftConnectInternalSeeds(training_set, objmap);
  iftDestroyLabeledSet(&training_set);
  //iftLabeledSet *seeds = training_set;


  /* to exchange across the three methods, change the comments
     below. You must also add the algorithm of the dynamic IFT using
     w5 as in the paper. */
  
  iftImage *label = NULL;
  label = iftDelineateObjectRegion(mimg,objmap,seeds,alpha);
  //label = iftDelineateObjectByWatershed(weight,seeds);
  //label = iftDelineateObjectByOrientedWatershed(weight,objmap,seeds);

  /* Draw segmentation border */
    
  iftDrawBorders(img, label, A, YCbCr, B);
  //iftMyDrawBinaryLabeledSeeds(img,seeds,YCbCr,A);

  iftWriteImageByExt(img,argv[4]);

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyAdjRel(&C);
  iftDestroyImage(&img);
  iftDestroyImage(&objmap);
  iftDestroyFImage(&weight);
  iftDestroyImage(&label);
  iftDestroyMImage(&mimg);
  iftDestroyLabeledSet(&seeds);
  
  return(0);
}

