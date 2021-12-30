#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct image_t
{
    int width;
    int height;
    int comp;
    int size_pixels;
    int size_bytes;
    uint8_t* DATA;
} image_t;

image_t* image_load(const char* _pathname, image_t** image);
void image_write(const char* _pathname, image_t** image);
void image_free(image_t** image);

image_t* image_load(const char* _pathname, image_t** image)
{
    if (*image == NULL)
    {
        *image = (image_t*)realloc(*image, sizeof(image_t));
    }

    printf("image: %s\n", _pathname);
    printf("begin image load...\n");

    (*image)->DATA = stbi_load(
        _pathname, &(*image)->width, &(*image)->height, &(*image)->comp, 0);

    assert((*image)->DATA != NULL);

    (*image)->size_pixels = (*image)->width * (*image)->height;
    (*image)->size_bytes = (*image)->size_pixels * (*image)->comp;

    printf("end image load...\n");
    printf("image is %dx%dpx, %d ch, %d pixels, %d bytes\n",
           (*image)->width,
           (*image)->height,
           (*image)->comp,
           (*image)->size_pixels,
           (*image)->size_bytes);

    return (*image);
}

void image_write(const char* _pathname, image_t** image)
{
    assert(*image != NULL);

    printf("image out: %s\n", _pathname);
    printf("begin image write...\n");

    int ret = stbi_write_png(_pathname,
                             (*image)->width,
                             (*image)->height,
                             (*image)->comp,
                             (*image)->DATA,
                             (*image)->width * (*image)->comp);

    if (ret < 0)
    {
        fprintf(stderr, "error writing image\n");
    }

    printf("end image write...\n");
    printf("image is %dx%dpx, %d ch, %d pixels, %d bytes\n",
           (*image)->width,
           (*image)->height,
           (*image)->comp,
           (*image)->size_pixels,
           (*image)->size_bytes);
}

void image_free(image_t** image)
{
    assert(*image != NULL);

    stbi_image_free((*image)->DATA);
    free(*image);
}
