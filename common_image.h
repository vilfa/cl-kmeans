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

image_t* image_load(const char* _pathname, image_t** image)
{
    if (*image == NULL)
    {
        *image = (image_t*)realloc(*image, sizeof(image_t));
    }

    printf("image: %s\n", _pathname);
    printf("begin image read...\n");

    (*image)->DATA = stbi_load(
        _pathname, &(*image)->width, &(*image)->height, &(*image)->comp, 0);

    if ((*image)->DATA == NULL)
    {
        fprintf(stderr, "error loading image\n");
    }

    (*image)->size_pixels = (*image)->width * (*image)->height;
    (*image)->size_bytes = (*image)->size_pixels * (*image)->comp;

    printf("end image read...\n");
    printf("image is %dx%dpx, %d ch\n",
           (*image)->width,
           (*image)->height,
           (*image)->comp);

    return (*image);
}

void image_write(const char* _pathname, image_t** image)
{
    assert(*image != NULL);

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
}

void image_free(image_t** image)
{
    assert(*image != NULL);

    stbi_image_free((*image)->DATA);
    free(*image);
}
