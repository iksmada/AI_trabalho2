#ifndef IFT_BRAINMSP_H_
#define IFT_BRAINMSP_H_

#ifdef __cplusplus
extern "C" {
#endif
  
#include "iftCommon.h"
#include "iftImage.h"
  
/**
 * @brief Detect the MSP and realign the image
 *
 * @author Guilherme C.S. Ruppert
 * @date Sep 17, 2017
 *
 * @param image The input image
 * @param mask An image mask of the brain (0=bg / 1=fg)  or NULL
 * @param input_ori The orientation: 1=sagittal / 2=axial / 3=coronal / 0=unknow
 * @param quality  1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
 *
 * @return A 3D volume realigned to the MSP
*/
  
 iftImage *iftMSP_Align(iftImage *image, iftImage *mask, int input_ori, int quality) ;
   
  


// Private
  


  
#ifdef __cplusplus
}
#endif

#endif
