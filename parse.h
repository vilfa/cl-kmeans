#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HELP \
    "USAGE:\n\
    compress [FLAGS] [OPTIONS]\n\
\n\
FLAGS:\n\
    -h, --help\n\
        Print help information.\n\
    -g\n\
        Use the GPU.\n\
    -x\n\
        No stdout.\n\
\n\
OPTIONS:\n\
    -i<IN_PATH>\n\
        Sets the input image path. Default: in.png.\n\
    -o<OUT_PATH>\n\
        Sets the output image path. Default: out.png.\n\
    -k<N_CENTROIDS>\n\
        Sets the number of centroids [2..256]. Default: 10.\n\
    -n<N_ITER>\n\
        Sets the iteration count [1..128]. Default: 16.\n\
    -t<N_THREADS>\n\
        Sets the thread count [1..64]. Default: 1.\n"

static int REQUIRED_ARGC = 1;
static char* DEFAULT_IMG_PATH_IN = "in.png";
static char* DEFAULT_IMG_PATH_OUT = "out.png";

typedef struct args_t
{
    char* img_path_in;
    char* img_path_out;
    int cluster_count;
    int iter_count;
    int thread_count;
    bool use_gpu;
    bool no_stdout;
} args_t;

args_t* args_init(args_t** args);
args_t* args_parse(args_t** args, int argc, const char** argv);
void args_free(args_t** args);

args_t* args_init(args_t** args)
{
    if (*args == NULL)
    {
        *args = (args_t*)realloc(*args, sizeof(args_t));
    }

    size_t len;

    len = strlen(DEFAULT_IMG_PATH_IN);
    (*args)->img_path_in = (char*)malloc((len + 1) * sizeof(char));
    memset((*args)->img_path_in, 0, len + 1);
    strcpy((*args)->img_path_in, DEFAULT_IMG_PATH_IN);

    len = strlen(DEFAULT_IMG_PATH_OUT);
    (*args)->img_path_out = (char*)malloc((len + 1) * sizeof(char));
    memset((*args)->img_path_out, 0, len + 1);
    strcpy((*args)->img_path_out, DEFAULT_IMG_PATH_OUT);

    (*args)->cluster_count = 10;
    (*args)->iter_count = 16;
    (*args)->thread_count = 1;
    (*args)->use_gpu = false;
    (*args)->no_stdout = false;

    return (*args);
}

args_t* args_parse(args_t** args, int argc, const char** argv)
{
    assert(args != NULL);
    assert(argc > REQUIRED_ARGC);

    if (argc == 2 &&
        (strncmp(argv[1], "-h", 2) == 0 || strncmp(argv[1], "--help", 6) == 0))
    {
        printf(HELP);
        args_free(args);
        exit(0);
    }

    const char* arg_names[] = {"-i", "-k", "-n", "-o", "-t", "-g", "-x"};

    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], arg_names[0], 2) == 0)
        {
            size_t len = strlen(argv[i] + 2);
            (*args)->img_path_in =
                (char*)realloc((*args)->img_path_in, len + 1);
            memset((*args)->img_path_in, 0, len + 1);
            strcpy((*args)->img_path_in, argv[i] + 2);
        }
        else if (strncmp(argv[i], arg_names[1], 2) == 0)
        {
            int val = atoi(argv[i] + 2);
            if (val < 2 || val > 256)
            {
                fprintf(
                    stderr,
                    "invalid cluster count: %d, should be between 2 and 256\n",
                    val);
            }
            else
            {
                (*args)->cluster_count = val;
            }
        }
        else if (strncmp(argv[i], arg_names[2], 2) == 0)
        {
            int val = atoi(argv[i] + 2);
            if (val < 1 || val > 128)
            {
                fprintf(stderr,
                        "invalid iteration count: %d, should be between 1 and "
                        "128\n",
                        val);
            }
            else
            {
                (*args)->iter_count = val;
            }
        }
        else if (strncmp(argv[i], arg_names[3], 2) == 0)
        {
            size_t len = strlen(argv[i] + 2);
            (*args)->img_path_out =
                (char*)realloc((*args)->img_path_out, len + 1);
            memset((*args)->img_path_out, 0, len + 1);
            strcpy((*args)->img_path_out, argv[i] + 2);
        }
        else if (strncmp(argv[i], arg_names[4], 2) == 0)
        {
            int val = atoi(argv[i] + 2);
            if (val < 1 || val > 64)
            {
                fprintf(
                    stderr,
                    "invalid thread count: %d, should be between 1 and 32\n",
                    val);
            }
            else
            {
                (*args)->thread_count = val;
            }
        }
        else if (strncmp(argv[i], arg_names[5], 2) == 0)
        {
            (*args)->use_gpu = true;
        }
        else if (strncmp(argv[i], arg_names[6], 2) == 0)
        {
            (*args)->no_stdout = true;
        }
        else
        {
            fprintf(stderr, "unknown argument at position: %d\n", i);
        }
    }

    printf(
        "running with arguments: "
        "img_in=%s,img_out=%s,k=%d,iter=%d,thr=%d,gpu=%d,no_stdout=%d\n",
        (*args)->img_path_in,
        (*args)->img_path_out,
        (*args)->cluster_count,
        (*args)->iter_count,
        (*args)->thread_count,
        (*args)->use_gpu,
        (*args)->no_stdout);

    return (*args);
}

void args_free(args_t** args)
{
    assert(args != NULL);

    free((*args)->img_path_in);
    free((*args)->img_path_out);
    free(*args);
}
