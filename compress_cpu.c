#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common_cl.h"
#include "common_files.h"
#include "common_image.h"

typedef struct kmean_sample_t
{
    uint8_t r;
    uint8_t g;
    uint8_t b;

} kmean_sample_t;

typedef struct kmean_t
{
    uint32_t k;
    uint32_t iter;
    kmean_sample_t* centroids;
} kmean_t;

kmean_t* kmeans_init(kmean_t** kmn, uint32_t k, uint32_t iter);
kmean_t* kmeans_cluster(kmean_t** kmn, image_t** image);
void kmeans_free(kmean_t** kmn);

int main()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    srand((uint32_t)ts.tv_nsec);

    image_t* image = NULL;

    load_image("images/doggo.jpg", &image);

    write_image("images/doggo2.png", &image);

    free_image(&image);

    kmean_t* kmeans = NULL;

    kmeans_init(&kmeans, 10, 100);

    kmeans_cluster(&kmeans, &image);

    kmeans_free(&kmeans);

    return 0;
}

kmean_t* kmeans_init(kmean_t** kmn, uint32_t k, uint32_t iter)
{
    if (*kmn == NULL)
    {
        *kmn = (kmean_t*)realloc(*kmn, sizeof(kmean_t));
    }

    (*kmn)->k = k;
    (*kmn)->iter = iter;
    (*kmn)->centroids = (kmean_sample_t*)malloc(k * sizeof(kmean_sample_t));

    return (*kmn);
}

kmean_t* kmeans_cluster(kmean_t** kmn, image_t** image)
{
    assert(*kmn != NULL);
    assert(*image != NULL);
    assert((*image)->DATA != NULL);

    uint32_t i;

    for (i = 0; i < (*kmn)->k; i++)
    {
        kmean_sample_t centroid;

        int ipx = rand() % ((*image)->width * (*image)->height);
        centroid.r = (*image)->DATA[ipx * (*image)->comp + 0];
        centroid.g = (*image)->DATA[ipx * (*image)->comp + 1];
        centroid.b = (*image)->DATA[ipx * (*image)->comp + 2];

        (*kmn)->centroids[i] =
    }

    i = 0;
    while (++i < (*kmn)->iter)
    {
    }
}

void kmeans_free(kmean_t** kmn)
{
    assert(*kmn != NULL);
    free((*kmn)->centroids);
    free(*kmn);
}
