#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "image.h"
#include "kmeans.h"
#include "ocl.h"
#include "parse.h"

int main(int argc, const char** argv)
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    srandom(ts.tv_nsec);

    args_t* args = NULL;
    image_t* image_in = NULL;
    image_t* image_out = NULL;
    kmean_t* kmeans = NULL;
    cl_env_t* clenv = NULL;

    args_init(&args);
    args_parse(&args, argc, argv);

    if (args->no_stdout)
    {
        fclose(stdout);
    }

    image_load(args->img_path_in, &image_in);
    kmeans_init(&kmeans, args->cluster_count, args->iter_count, &image_in);

    if (args->use_gpu)
    {
        cl_init(&clenv);
        kmeans_cluster_gpu(&kmeans, &clenv, &image_in, &image_out);
        image_write(args->img_path_out, &image_out);
        cl_free(&clenv);
    }
    else if (args->thread_count > 1)
    {
        kmeans_cluster_multithr(&kmeans, &image_in, args->thread_count);
        kmeans_image_multithr(
            &kmeans, &image_in, &image_out, args->thread_count);
        image_write(args->img_path_out, &image_out);
    }
    else
    {
        kmeans_cluster(&kmeans, &image_in);
        kmeans_image(&kmeans, &image_in, &image_out);
        image_write(args->img_path_out, &image_out);
    }

    args_free(&args);
    kmeans_free(&kmeans);
    image_free(&image_in);
    image_free(&image_out);

    return 0;
}
