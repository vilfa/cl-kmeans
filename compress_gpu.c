#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "files.h"
#include "image.h"
#include "kmeans.h"
#include "ocl.h"
#include "parse.h"

int main(int argc, const char** argv)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    srand((uint32_t)ts.tv_nsec);

    args_t* args = NULL;
    image_t* image = NULL;
    kmean_t* kmeans = NULL;
    cl_env_t* clenv = NULL;
    image_t* image_out = NULL;

    args_init(&args);

    args_parse(&args, argc, argv);

    image_load(args->imgpath, &image);

    kmeans_init(&kmeans, args->cluster_count, args->iter_count, &image);

    cl_init(&clenv);

    kmeans_cluster_gpu(&kmeans, &clenv, &image, &image_out);

    args_free(&args);

    kmeans_free(&kmeans);

    image_free(&image);

    image_free(&image_out);
}
