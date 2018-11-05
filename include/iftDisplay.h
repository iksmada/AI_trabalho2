//
// Created by jookuma on 27/12/17.
//

#ifndef IFT_IFTDISPLAY_H
#define IFT_IFTDISPLAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftImage.h"
#include "iftColor.h"
#include "iftUtil.h"

#ifdef __linux__
#include "SDL.h"
#endif

typedef struct ift_image_display {
#ifdef __linux__
    /* SDL2 Objects */
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
#else
    /* Only so other functions don't crash */
    void *window;
    void *renderer;
    void *texture;
#endif
    /* Length of row in memory (xsize * bands * sizeof(u_int8_t) */
    int pitch;
    /* Number of pixels */
    int n;
    /* Buffer of rgb image to display, very different from
     * our image format, check https://en.wikipedia.org/wiki/RGBA_color_space */
    u_int8_t *buffer;
    u_int8_t *orig;
} iftImageDisplay;

/**
 * @brief Allocs objects for ift interface with SDL2 Library
 * @author Jordão Bragantini
 * @param YCbCr Image
 * @return struct for image display functions
 */
iftImageDisplay *iftCreateImageDisplay(iftImage* ycbcr_img);

/**
 * @brief Convert to RGB and copy image data to image display buffer;
 * @author Jordão Bragantini
 * @param ImageDisplay object
 * @param YCbCr Image to be copied
 */
void iftUpdateImageDisplayBuffer(iftImageDisplay *img_display, iftImage *img);

/**
 * @brief Copy rgb color object to img_display buffer on index of pixel p;
 * @author Jordão Bragantini
 * @param ImageDisplay object
 * @param pixel index p
 * @param color object rgb
 */
void iftUpdateImageDisplayBufferPixelColor(iftImageDisplay *img_display, int p, iftColor *rgb);

/**
 * @brief Update image display rendering;
 * @author Jordão Bragantini
 * @param ImageDisplay object
 */
void iftUpdateImageDisplayRender(iftImageDisplay *img_display);

/**
 * @brief Free ImageDisplay object;
 * @author Jordão Bragantini
 * @param ImageDisplay object
 */
void iftDestroyImageDisplay(iftImageDisplay **img_display);

/**
 * @brief Copy original image to buffer;
 * @author Jordão Bragantini
 * @param ImageDisplay object
 */
void iftResetImageDisplayBuffer(iftImageDisplay *img_display);

/**
 * @brief Set a percentage of the original image saturation to a given pixel or the whole image
 *        ATENTION: Very slow function because of the rendering.
 * @author Jordão Bragantini
 * @param ImageDisplay object
 * @param index of pixel that will be multiplied, if its -1 it will apply for the whole buffer
 * @param percentage of the original image that will be set on that index
 */
void iftSetImageDisplaySaturation(iftImageDisplay *img_display, int index, float percent);

#endif //IFT_IFTDISPLAY_H
