#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common_cl.h"
#include "common_files.h"
#include "common_image.h"

typedef struct kmean_sample_t
{
    int r;
    int g;
    int b;
    double norm;
} kmean_sample_t;

typedef struct kmean_t
{
    uint32_t k;
    uint32_t iter;
    uint32_t* px_centroid;
    kmean_sample_t* centroids;
} kmean_t;

kmean_t* kmeans_init(kmean_t** kmn, uint32_t k, uint32_t iter, image_t** img);
kmean_t* kmeans_cluster(kmean_t** kmn, image_t** img);
void kmeans_free(kmean_t** kmn);

double sample_norm(kmean_sample_t* sample);
double sample_euclid2(kmean_sample_t* sample1, kmean_sample_t* sample2);

int main()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    srand((uint32_t)ts.tv_nsec);

    image_t* image = NULL;

    image_load("images/doggo.jpg", &image);

    kmean_t* kmeans = NULL;

    kmeans_init(&kmeans, 10, 16, &image);

    kmeans_cluster(&kmeans, &image);

    kmeans_free(&kmeans);

    image_free(&image);

    return 0;
}

kmean_t* kmeans_init(kmean_t** kmn, uint32_t k, uint32_t iter, image_t** img)
{
    assert(*img != NULL);

    if (*kmn == NULL)
    {
        *kmn = (kmean_t*)realloc(*kmn, sizeof(kmean_t));
    }

    (*kmn)->k = k;
    (*kmn)->iter = iter;
    (*kmn)->centroids = (kmean_sample_t*)malloc(k * sizeof(kmean_sample_t));
    (*kmn)->px_centroid =
        (uint32_t*)malloc((*img)->size_pixels * sizeof(uint32_t));

    printf("initialize cluster kmeans clustering...\n");
    printf("cluster count is %d, iteration count is %d\n",
           (*kmn)->k,
           (*kmn)->iter);

    return (*kmn);
}

kmean_t* kmeans_cluster(kmean_t** kmn, image_t** img)
{
    assert(*kmn != NULL);
    assert(*img != NULL);

    printf("begin clustering...\n");

    for (uint32_t k = 0; k < (*kmn)->k; k++)
    {
        kmean_sample_t centroid;

        size_t ipx = rand() % ((*img)->size_pixels);

        centroid.r = (int)((*img)->DATA[ipx * (*img)->comp + 0]);
        centroid.g = (int)((*img)->DATA[ipx * (*img)->comp + 1]);
        centroid.b = (int)((*img)->DATA[ipx * (*img)->comp + 2]);
        centroid.norm = sample_norm(&centroid);

        (*kmn)->centroids[k] = centroid;

        printf("c%d: %d, %d, %d, norm %f\n",
               k,
               centroid.r,
               centroid.g,
               centroid.b,
               centroid.norm);
    }

    uint32_t iter = 0;
    while (++iter < (*kmn)->iter)
    {
        kmean_sample_t sample;

        // Iterate through each pixel in image.
        for (uint32_t i = 0; i < (*img)->size_pixels; i++)
        {
            sample.r = (int)((*img)->DATA[i * (*img)->comp + 0]);
            sample.g = (int)((*img)->DATA[i * (*img)->comp + 1]);
            sample.b = (int)((*img)->DATA[i * (*img)->comp + 2]);

            double euclid = DBL_MAX;
            uint32_t kgroup = 0;
            // Iterate through each group of k groups.
            for (uint32_t k = 0; k < (*kmn)->k; k++)
            {
                // Find the smallest euclid distance to a centroid for this
                // pixel.
                double e = sample_euclid2(&((*kmn)->centroids[k]), &sample);
                if (e < euclid)
                {
                    euclid = e;
                    kgroup = k;
                }
            }

            // This pixel now belongs to the group with the nearest centroid.
            (*kmn)->px_centroid[i] = kgroup;
        }

        // Calculate the new centroid for each pixel group.
        for (uint32_t k = 0; k < (*kmn)->k; k++)
        {
            int r = 0;
            int g = 0;
            int b = 0;
            uint32_t group_size = 0;

            for (uint32_t i = 0; i < (*img)->size_pixels; i++)
            {
                // Add up all values for pixels in a certain cluster
                if ((*kmn)->px_centroid[i] == k)
                {
                    group_size++;
                    r += (int)((*img)->DATA[i * (*img)->comp + 0]);
                    g += (int)((*img)->DATA[i * (*img)->comp + 1]);
                    b += (int)((*img)->DATA[i * (*img)->comp + 2]);
                }
            }

            // Make sure we don't get a SIGFPE with division by zero.
            if (group_size == 0) continue;

            // Average out all the pixel values.
            (*kmn)->centroids[k].r = r / group_size;
            (*kmn)->centroids[k].g = g / group_size;
            (*kmn)->centroids[k].b = b / group_size;
            (*kmn)->centroids[k].norm = sample_norm((&(*kmn)->centroids[k]));
        }
    }

    printf("end clustering...\n");

    for (uint32_t k = 0; k < (*kmn)->k; k++)
    {
        printf("c%d: %d, %d, %d, norm %f\n",
               k,
               (*kmn)->centroids[k].r,
               (*kmn)->centroids[k].g,
               (*kmn)->centroids[k].b,
               (*kmn)->centroids[k].norm);
    }

    return (*kmn);
}

void kmeans_free(kmean_t** kmn)
{
    assert(*kmn != NULL);
    free((*kmn)->centroids);
    free((*kmn)->px_centroid);
    free(*kmn);
}

double sample_norm(kmean_sample_t* sample)
{
    assert(sample != NULL);

    return sqrt(pow((double)sample->r, 2) + pow((double)sample->g, 2) +
                pow((double)sample->b, 2));
}

double sample_euclid2(kmean_sample_t* sample1, kmean_sample_t* sample2)
{
    assert(sample1 != NULL);
    assert(sample2 != NULL);

    int r = sample1->r - sample2->r;
    int g = sample1->g - sample2->g;
    int b = sample1->b - sample2->b;

    double euclid =
        sqrt(pow((double)r, 2) + pow((double)g, 2) + pow((double)b, 2));

    return pow(euclid, 2);
}
