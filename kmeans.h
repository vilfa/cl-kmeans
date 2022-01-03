#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "files.h"
#include "image.h"
#include "ocl.h"

typedef struct kmean_sample_t
{
    int r;
    int g;
    int b;
} kmean_sample_t;

typedef struct kmean_t
{
    int k;
    int iter;
    int* px_centroid;
    kmean_sample_t* centroids;
} kmean_t;

kmean_t* kmeans_init(kmean_t** kmn, int k, int iter, image_t** img);
kmean_t* kmeans_cluster(kmean_t** kmn, image_t** img);
kmean_t* kmeans_cluster_multithr(kmean_t** kmn, image_t** img_in, int threads);
kmean_t* kmeans_cluster_gpu(kmean_t** kmn,
                            cl_env_t** env,
                            image_t** img_in,
                            image_t** img_out);
kmean_t* kmeans_image(kmean_t** kmn, image_t** img_in, image_t** img_out);
kmean_t* kmeans_image_multithr(kmean_t** kmn,
                               image_t** img_in,
                               image_t** img_out,
                               int threads);
double kmeans_sample_norm(kmean_sample_t* sample);
double kmeans_sample_euclid2(kmean_sample_t* sample1, kmean_sample_t* sample2);
void kmeans_free(kmean_t** kmn);

kmean_t* kmeans_init(kmean_t** kmn, int k, int iter, image_t** img)
{
    assert(*img != NULL);

    if (*kmn == NULL)
    {
        *kmn = (kmean_t*)realloc(*kmn, sizeof(kmean_t));
    }

    (*kmn)->k = k;
    (*kmn)->iter = iter;
    (*kmn)->centroids = (kmean_sample_t*)malloc(k * sizeof(kmean_sample_t));
    (*kmn)->px_centroid = (int*)malloc((*img)->size_pixels * sizeof(int));

    printf("initialize kmeans clustering...\n");
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

    for (int k = 0; k < (*kmn)->k; k++)
    {
        kmean_sample_t centroid;

        int i = rand() % ((*img)->size_pixels);

        centroid.r = (int)((*img)->DATA[i * (*img)->comp + 0]);
        centroid.g = (int)((*img)->DATA[i * (*img)->comp + 1]);
        centroid.b = (int)((*img)->DATA[i * (*img)->comp + 2]);

        (*kmn)->centroids[k] = centroid;

        printf("c%d: %d, %d, %d\n", k, centroid.r, centroid.g, centroid.b);
    }

    int iter = 0;
    int* group_size = (int*)calloc((*kmn)->k, sizeof(int));
    int* rgb_vals = (int*)calloc(3 * (*kmn)->k, sizeof(int));
    while (iter++ < (*kmn)->iter)
    {
        printf("processing iteration %d/%d...\n", iter, (*kmn)->iter);

        kmean_sample_t sample;

        // Iterate through each pixel in image.
        double euclid;
        int group;
        for (int i = 0; i < (*img)->size_pixels; i++)
        {
            sample.r = (int)((*img)->DATA[i * (*img)->comp + 0]);
            sample.g = (int)((*img)->DATA[i * (*img)->comp + 1]);
            sample.b = (int)((*img)->DATA[i * (*img)->comp + 2]);

            euclid = DBL_MAX;
            group = 0;
            // Iterate through each group of k groups.
            for (int k = 0; k < (*kmn)->k; k++)
            {
                // Find the smallest squared euclid distance to a centroid for
                // this pixel.
                double e =
                    kmeans_sample_euclid2(&((*kmn)->centroids[k]), &sample);
                if (e < euclid)
                {
                    euclid = e;
                    group = k;
                }
            }

            // This pixel now belongs to the group with the nearest centroid.
            (*kmn)->px_centroid[i] = group;
        }

        // Calculate the new centroid for each pixel group.
        memset(group_size, 0, (*kmn)->k * sizeof(int));
        memset(rgb_vals, 0, 3 * (*kmn)->k * sizeof(int));
        for (int i = 0; i < (*img)->size_pixels; i++)
        {
            group_size[(*kmn)->px_centroid[i]]++;
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 0] +=
                (int)((*img)->DATA[i * (*img)->comp + 0]);
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 1] +=
                (int)((*img)->DATA[i * (*img)->comp + 1]);
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 2] +=
                (int)((*img)->DATA[i * (*img)->comp + 2]);
        }

        // Average out all the pixel values.
        for (int k = 0; k < (*kmn)->k; k++)
        {
            if (group_size[k] == 0) continue;
            (*kmn)->centroids[k].r = rgb_vals[k * 3 + 0] / group_size[k];
            (*kmn)->centroids[k].g = rgb_vals[k * 3 + 1] / group_size[k];
            (*kmn)->centroids[k].b = rgb_vals[k * 3 + 2] / group_size[k];
        }
    }

    free(group_size);
    free(rgb_vals);
    printf("end clustering...\n");

    for (int k = 0; k < (*kmn)->k; k++)
    {
        printf("c%d: %d, %d, %d\n",
               k,
               (*kmn)->centroids[k].r,
               (*kmn)->centroids[k].g,
               (*kmn)->centroids[k].b);
    }

    return (*kmn);
}

kmean_t* kmeans_cluster_gpu(kmean_t** kmn,
                            cl_env_t** env,
                            image_t** img_in,
                            image_t** img_out)
{
    assert(*kmn != NULL);
    assert(*env != NULL);
    assert(*img_in != NULL);

    int* rand_vector = (int*)malloc((*kmn)->k * sizeof(int));
    for (int k = 0; k < (*kmn)->k; k++)
    {
        rand_vector[k] = rand() % ((*img_in)->size_pixels);
    }

    char* buf = NULL;
    file_read("compress.cl", &buf, BUFSIZ);

    cl_program* program = cl_create_program(env, buf);
    cl_xpair_t* xpair = cl_create_kernel(env, program, "compress");

    cl_mem img_in_mem_obj =
        clCreateBuffer((*env)->context,
                       CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                       (*img_in)->size_bytes,
                       (void*)((*img_in)->DATA),
                       &CL_RET);
    CL_CHECK_ERR(CL_RET);

    cl_mem kmeans_rand_vector_mem_obj =
        clCreateBuffer((*env)->context,
                       CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                       (*kmn)->k * sizeof(int),
                       (void*)rand_vector,
                       &CL_RET);
    CL_CHECK_ERR(CL_RET);

    cl_mem kmeans_centroids_mem_obj =
        clCreateBuffer((*env)->context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       3 * (*kmn)->k * sizeof(int),
                       NULL,
                       &CL_RET);
    CL_CHECK_ERR(CL_RET);

    cl_mem kmeans_px_centroids_mem_obj =
        clCreateBuffer((*env)->context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       (*img_in)->size_pixels * sizeof(int),
                       NULL,
                       &CL_RET);
    CL_CHECK_ERR(CL_RET);

    cl_mem kmeans_group_size_mem_obj =
        clCreateBuffer((*env)->context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       (*kmn)->k * sizeof(int),
                       NULL,
                       &CL_RET);
    CL_CHECK_ERR(CL_RET);

    cl_mem kmeans_rgb_values_mem_obj =
        clCreateBuffer((*env)->context,
                       CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                       3 * (*kmn)->k * sizeof(int),
                       NULL,
                       &CL_RET);
    CL_CHECK_ERR(CL_RET);

    cl_add_kernel_arg_mem_obj(env, xpair, 0, sizeof(cl_mem), img_in_mem_obj);
    cl_add_kernel_arg_mem_obj(
        env, xpair, 1, sizeof(cl_mem), kmeans_rand_vector_mem_obj);
    cl_add_kernel_arg_mem_obj(
        env, xpair, 2, sizeof(cl_mem), kmeans_centroids_mem_obj);
    cl_add_kernel_arg_mem_obj(
        env, xpair, 3, sizeof(cl_mem), kmeans_px_centroids_mem_obj);
    cl_add_kernel_arg_mem_obj(
        env, xpair, 4, sizeof(cl_mem), kmeans_group_size_mem_obj);
    cl_add_kernel_arg_mem_obj(
        env, xpair, 5, sizeof(cl_mem), kmeans_rgb_values_mem_obj);
    cl_add_kernel_arg_prim(env, xpair, 6, sizeof(int), (void*)&((*kmn)->k));
    cl_add_kernel_arg_prim(env, xpair, 7, sizeof(int), (void*)&((*kmn)->iter));
    cl_add_kernel_arg_prim(
        env, xpair, 8, sizeof(int), (void*)&((*img_in)->width));
    cl_add_kernel_arg_prim(
        env, xpair, 9, sizeof(int), (void*)&((*img_in)->height));
    cl_add_kernel_arg_prim(
        env, xpair, 10, sizeof(int), (void*)&((*img_in)->comp));

    const size_t _local_work_size = 512;
    const size_t _workgroup_count = (*img_in)->size_pixels / _local_work_size;
    const size_t _global_work_size = _local_work_size * _workgroup_count;

    cl_enqueue_kernel(
        env, xpair, 1, &_global_work_size, &_local_work_size, NULL);

    int* centroids = (int*)malloc(3 * (*kmn)->k * sizeof(int));
    int* px_centroids = (int*)malloc((*img_in)->size_pixels * sizeof(int));

    cl_read_buffer(env,
                   &kmeans_centroids_mem_obj,
                   CL_TRUE,
                   3 * (*kmn)->k * sizeof(int),
                   (void*)centroids);

    cl_read_buffer(env,
                   &kmeans_px_centroids_mem_obj,
                   CL_TRUE,
                   (*img_in)->size_pixels * sizeof(int),
                   (void*)px_centroids);

    for (int i = 0; i < (*kmn)->k; i++)
    {
        printf("c%d: %d, %d, %d\n",
               i,
               centroids[i * 3 + 0],
               centroids[i * 3 + 1],
               centroids[i * 3 + 2]);

        (*kmn)->centroids[i].r = centroids[i * 3 + 0];
        (*kmn)->centroids[i].g = centroids[i * 3 + 1];
        (*kmn)->centroids[i].b = centroids[i * 3 + 2];
    }

    memcpy((*kmn)->px_centroid,
           px_centroids,
           (*img_in)->size_pixels * sizeof(int));

    for (int i = 0; i < (*img_in)->size_pixels / 10000; i++)
    {
        printf("%d ", px_centroids[i]);
    }
    printf("\n");

    free(centroids);
    free(px_centroids);

    kmeans_image(kmn, img_in, img_out);

    return (*kmn);
}

kmean_t* kmeans_cluster_multithr(kmean_t** kmn, image_t** img, int threads)
{
    assert(*kmn != NULL);
    assert(*img != NULL);

    omp_set_num_threads(threads);

    int* group_size = (int*)calloc((*kmn)->k, sizeof(int));
    int* rgb_vals = (int*)calloc(3 * (*kmn)->k, sizeof(int));

    printf("begin clustering with %d threads...\n", threads);

#pragma omp parallel for schedule(dynamic) shared(kmn, img) default(none)
    for (int k = 0; k < (*kmn)->k; k++)
    {
        kmean_sample_t centroid;

        int i = rand() % ((*img)->size_pixels);

        centroid.r = (int)((*img)->DATA[i * (*img)->comp + 0]);
        centroid.g = (int)((*img)->DATA[i * (*img)->comp + 1]);
        centroid.b = (int)((*img)->DATA[i * (*img)->comp + 2]);

#pragma omp critical
        (*kmn)->centroids[k] = centroid;

        printf("c%d: %d, %d, %d\n", k, centroid.r, centroid.g, centroid.b);
    }

    int iter = 0;
    while (iter++ < (*kmn)->iter)
    {
        printf("processing iteration %d/%d...\n", iter, (*kmn)->iter);

        kmean_sample_t sample;

        // Iterate through each pixel in image.
        double euclid;
        int group;
#pragma omp parallel for schedule(dynamic) private(euclid, group, sample) \
    shared(img, kmn) default(none)
        for (int i = 0; i < (*img)->size_pixels; i++)
        {
            sample.r = (int)((*img)->DATA[i * (*img)->comp + 0]);
            sample.g = (int)((*img)->DATA[i * (*img)->comp + 1]);
            sample.b = (int)((*img)->DATA[i * (*img)->comp + 2]);

            euclid = DBL_MAX;
            group = 0;
            // Iterate through each group of k groups.
            for (int k = 0; k < (*kmn)->k; k++)
            {
                // Find the smallest squared euclid distance to a centroid
                // for this pixel.
                double e =
                    kmeans_sample_euclid2(&((*kmn)->centroids[k]), &sample);
                if (e < euclid)
                {
                    euclid = e;
                    group = k;
                }
            }

            // This pixel now belongs to the group with the nearest
            // centroid.
            (*kmn)->px_centroid[i] = group;
        }

#pragma omp barrier

        // Calculate the new centroid for each pixel group
#pragma omp parallel for schedule(dynamic) \
    shared(group_size, rgb_vals, img, kmn) default(none)
        for (int i = 0; i < (*img)->size_pixels; i++)
        {
#pragma omp atomic
            group_size[(*kmn)->px_centroid[i]]++;
#pragma omp atomic
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 0] +=
                (int)((*img)->DATA[i * (*img)->comp + 0]);
#pragma omp atomic
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 1] +=
                (int)((*img)->DATA[i * (*img)->comp + 1]);
#pragma omp atomic
            rgb_vals[(*kmn)->px_centroid[i] * 3 + 2] +=
                (int)((*img)->DATA[i * (*img)->comp + 2]);
        }

#pragma omp barrier

// Average out all the pixel values.
#pragma omp parallel for schedule(dynamic) \
    shared(kmn, group_size, rgb_vals) default(none)
        for (int k = 0; k < (*kmn)->k; k++)
        {
            if (group_size[k] == 0) continue;
            (*kmn)->centroids[k].r = rgb_vals[k * 3 + 0] / group_size[k];
            (*kmn)->centroids[k].g = rgb_vals[k * 3 + 1] / group_size[k];
            (*kmn)->centroids[k].b = rgb_vals[k * 3 + 2] / group_size[k];
        }
    }

    free(group_size);
    free(rgb_vals);

    printf("end clustering...\n");

    for (int k = 0; k < (*kmn)->k; k++)
    {
        printf("c%d: %d, %d, %d\n",
               k,
               (*kmn)->centroids[k].r,
               (*kmn)->centroids[k].g,
               (*kmn)->centroids[k].b);
    }

    return (*kmn);
}

kmean_t* kmeans_image_multithr(kmean_t** kmn,
                               image_t** img_in,
                               image_t** img_out,
                               int threads)
{
    assert(*kmn != NULL);
    assert(*img_in != NULL);

    printf("writing image data with %d threads...\n", threads);

    omp_set_num_threads(threads);

    if (*img_out == NULL)
    {
        *img_out = (image_t*)realloc(*img_out, sizeof(image_t));
    }

    (*img_out)->DATA = (uint8_t*)malloc((*img_in)->width * (*img_in)->height *
                                        4 * sizeof(uint8_t));

    (*img_out)->width = (*img_in)->width;
    (*img_out)->height = (*img_in)->height;
    (*img_out)->comp = 4;
    (*img_out)->size_pixels = (*img_in)->size_pixels;
    (*img_out)->size_bytes = (*img_in)->size_bytes;

#pragma omp parallel for schedule(dynamic) \
    shared(img_in, img_out, kmn) default(none)
    for (int i = 0; i < (*img_in)->size_pixels; i++)
    {
        (*img_out)->DATA[i * 4 + 0] =
            (*kmn)->centroids[(*kmn)->px_centroid[i]].r;
        (*img_out)->DATA[i * 4 + 1] =
            (*kmn)->centroids[(*kmn)->px_centroid[i]].g;
        (*img_out)->DATA[i * 4 + 2] =
            (*kmn)->centroids[(*kmn)->px_centroid[i]].b;
        (*img_out)->DATA[i * 4 + 3] = 255;
    }

    return (*kmn);
}

kmean_t* kmeans_image(kmean_t** kmn, image_t** img_in, image_t** img_out)
{
    assert(*kmn != NULL);
    assert(*img_in != NULL);

    printf("writing image data...\n");

    if (*img_out == NULL)
    {
        *img_out = (image_t*)realloc(*img_out, sizeof(image_t));
    }

    (*img_out)->DATA = (uint8_t*)malloc((*img_in)->width * (*img_in)->height *
                                        4 * sizeof(uint8_t));

    (*img_out)->width = (*img_in)->width;
    (*img_out)->height = (*img_in)->height;
    (*img_out)->comp = 4;
    (*img_out)->size_pixels = (*img_in)->size_pixels;
    (*img_out)->size_bytes = (*img_in)->size_bytes;

    for (int i = 0; i < (*img_in)->size_pixels; i++)
    {
        (*img_out)->DATA[i * 4 + 0] =
            (*kmn)->centroids[(*kmn)->px_centroid[i]].r;
        (*img_out)->DATA[i * 4 + 1] =
            (*kmn)->centroids[(*kmn)->px_centroid[i]].g;
        (*img_out)->DATA[i * 4 + 2] =
            (*kmn)->centroids[(*kmn)->px_centroid[i]].b;
        (*img_out)->DATA[i * 4 + 3] = 255;
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

double kmeans_sample_norm(kmean_sample_t* sample)
{
    assert(sample != NULL);

    return sqrt(pow((double)sample->r, 2) + pow((double)sample->g, 2) +
                pow((double)sample->b, 2));
}

double kmeans_sample_euclid2(kmean_sample_t* sample1, kmean_sample_t* sample2)
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
