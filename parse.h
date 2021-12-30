#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int REQUIRED_ARGC = 1;
static char* DEFAULT_IMGPATH = "images/image.png";
static char* DEFAULT_IMGPATH_OUT = "images/out.png";

typedef struct args_t
{
    char* imgpath;
    char* imgpath_out;
    int cluster_count;
    int iter_count;
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

    int len;

    len = strlen(DEFAULT_IMGPATH);
    (*args)->imgpath = (char*)malloc((len + 1) * sizeof(char));
    memset((*args)->imgpath, 0, len + 1);
    strcpy((*args)->imgpath, DEFAULT_IMGPATH);

    len = strlen(DEFAULT_IMGPATH_OUT);
    (*args)->imgpath_out = (char*)malloc((len + 1) * sizeof(char));
    memset((*args)->imgpath_out, 0, len + 1);
    strcpy((*args)->imgpath_out, DEFAULT_IMGPATH_OUT);

    (*args)->cluster_count = 10;
    (*args)->iter_count = 16;

    return (*args);
}

args_t* args_parse(args_t** args, int argc, const char** argv)
{
    assert(args != NULL);
    assert(argc > REQUIRED_ARGC);

    const char* arg_names[] = {"-i", "-k", "-n", "-o"};

    for (int i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], arg_names[0], 2) == 0)
        {
            size_t len = strlen(argv[i] + 2);
            (*args)->imgpath = (char*)realloc((*args)->imgpath, len + 1);
            memset((*args)->imgpath, 0, len + 1);
            strcpy((*args)->imgpath, argv[i] + 2);
        }
        else if (strncmp(argv[i], arg_names[1], 2) == 0)
        {
            int val = atoi(argv[i] + 2);
            if (val < 2 || val > 64)
            {
                fprintf(
                    stderr,
                    "invalid cluster count: %d, should be between 2 and 64\n",
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
            if (val < 1 || val > 100)
            {
                fprintf(stderr,
                        "invalid iteration count: %d, should be between 1 and "
                        "100\n",
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
            (*args)->imgpath_out =
                (char*)realloc((*args)->imgpath_out, len + 1);
            memset((*args)->imgpath_out, 0, len + 1);
            strcpy((*args)->imgpath_out, argv[i] + 2);
        }
        else
        {
            fprintf(stderr, "unknown argument at position: %d\n", i);
        }
    }

    printf("running with arguments: img_in=%s,img_out=%s,k=%d,iter=%d\n",
           (*args)->imgpath,
           (*args)->imgpath_out,
           (*args)->cluster_count,
           (*args)->iter_count);

    return (*args);
}

void args_free(args_t** args)
{
    assert(args != NULL);

    free((*args)->imgpath);
    free((*args)->imgpath_out);
    free(*args);
}
