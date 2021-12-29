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
    uint8_t* DATA;
} image_t;

image_t* load_image(const char* _pathname, image_t** image)
{
    if (*image == NULL)
    {
        *image = (image_t*)realloc(*image, sizeof(image_t));
    }

    (*image)->DATA = stbi_load(
        _pathname, &(*image)->width, &(*image)->height, &(*image)->comp, 0);

    if ((*image)->DATA == NULL)
    {
        fprintf(stderr, "Error reading image\n");
    }

    return (*image);
}

void write_image(const char* _pathname, image_t** image)
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
        fprintf(stderr, "Error writing image\n");
    }
}

void free_image(image_t** image)
{
    assert(*image != NULL);

    stbi_image_free((*image)->DATA);
    free(*image);
}
