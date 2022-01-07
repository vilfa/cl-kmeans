#pragma once

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_CHECK_ERR(ret)                       \
    if ((ret) < 0)                                \
    fprintf(stderr,                             \
            "%s, line %d: cl error (%d): %s\n", \
            __FILE__,                           \
            __LINE__,                           \
            ret,                                \
            cl_error_string(ret))

static cl_int CL_RET;

typedef struct cl_xpair_t
{
    cl_program program;
    cl_kernel kernel;

    int kernel_arg_count;
    int kernel_arg_prim_count;
    int kernel_arg_mem_obj_count;
    cl_mem* kernel_arg_mem_objs;
} cl_xpair_t;

typedef struct cl_env_t
{
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;

    int program_count;
    int xpair_count;
    cl_program* programs;
    cl_xpair_t* xpairs;
} cl_env_t;

cl_env_t* cl_init(cl_env_t** env);
cl_program* cl_create_program(cl_env_t** env, const char* _source);
cl_xpair_t* cl_create_kernel(cl_env_t** env,
                             cl_program* program,
                             const char* _name);
cl_xpair_t* cl_add_kernel_arg_mem_obj(cl_env_t** env,
                                      cl_xpair_t* execution_pair,
                                      cl_uint position,
                                      size_t size,
                                      cl_mem mem_obj);
cl_xpair_t* cl_add_kernel_arg_prim(cl_env_t** env,
                                   cl_xpair_t* execution_pair,
                                   cl_uint position,
                                   size_t size,
                                   const void* _ptr);
cl_xpair_t* cl_enqueue_kernel(cl_env_t** env,
                              cl_xpair_t* execution_pair,
                              cl_uint work_dimensions,
                              const size_t* _global_work_size,
                              const size_t* _local_work_size,
                              cl_event* event);
void* cl_read_buffer(cl_env_t** env,
                     cl_mem* source_mem_obj,
                     cl_bool blocking_read,
                     size_t size,
                     void* ptr);
void cl_free(cl_env_t** env);
const char* cl_error_string(cl_int err);

cl_env_t* cl_init(cl_env_t** env)
{
    if (*env == NULL)
    {
        *env = (cl_env_t*)realloc(*env, sizeof(cl_env_t));
    }

    (*env)->program_count = 0;
    (*env)->programs = NULL;
    (*env)->xpair_count = 0;
    (*env)->xpairs = NULL;

    cl_platform_id platform_id;
    cl_uint ret_num_platforms;
    CL_RET = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    CL_CHECK_ERR(CL_RET);
    assert(ret_num_platforms > 0);

    (*env)->platform_id = platform_id;

    cl_device_id device_id;
    cl_uint ret_num_devices;
    CL_RET = clGetDeviceIDs(
        platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    CL_CHECK_ERR(CL_RET);
    assert(ret_num_devices > 0);

    (*env)->device_id = device_id;

    cl_context context =
        clCreateContext(NULL, 1, &device_id, NULL, NULL, &CL_RET);
    CL_CHECK_ERR(CL_RET);

    (*env)->context = context;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, &CL_RET);
#pragma GCC diagnostic pop
    CL_CHECK_ERR(CL_RET);

    (*env)->command_queue = command_queue;

    printf("initialized cl environment, platforms=%d gpu_devices=%d\n",
           ret_num_platforms,
           ret_num_devices);

    return *env;
}

cl_program* cl_create_program(cl_env_t** env, const char* _source)
{
    assert(*env != NULL);
    assert(_source != NULL);

    cl_program program =
        clCreateProgramWithSource((*env)->context, 1, &_source, NULL, &CL_RET);
    CL_CHECK_ERR(CL_RET);

    printf("created program...\n");

    size_t ret_build_log_size = 0;
    char* build_log = NULL;

    CL_RET = clBuildProgram(program, 1, &(*env)->device_id, NULL, NULL, NULL);
    CL_CHECK_ERR(CL_RET);

    printf("built program...\n");

    CL_RET = clGetProgramBuildInfo(program,
                                   (*env)->device_id,
                                   CL_PROGRAM_BUILD_LOG,
                                   0,
                                   NULL,
                                   &ret_build_log_size);
    CL_CHECK_ERR(CL_RET);

    build_log = (char*)realloc(build_log, ret_build_log_size * sizeof(char));
    CL_RET = clGetProgramBuildInfo(program,
                                   (*env)->device_id,
                                   CL_PROGRAM_BUILD_LOG,
                                   ret_build_log_size,
                                   build_log,
                                   NULL);
    CL_CHECK_ERR(CL_RET);

    if (ret_build_log_size > 1)
        fprintf(stderr, "program build log:\n%s\n", build_log);
    free(build_log);

    (*env)->program_count++;
    (*env)->programs = (cl_program*)realloc(
        (*env)->programs, (*env)->program_count * sizeof(cl_program));

    (*env)->programs[(*env)->program_count - 1] = program;

    printf("created cl program, count is now %d\n", (*env)->program_count);

    return &((*env)->programs[(*env)->program_count - 1]);
}

cl_xpair_t* cl_create_kernel(cl_env_t** env,
                             cl_program* program,
                             const char* _name)
{
    assert(*env != NULL);
    assert(program != NULL);

    cl_kernel kernel = clCreateKernel(*program, _name, &CL_RET);
    CL_CHECK_ERR(CL_RET);

    printf("created kernel...\n");

    cl_xpair_t execution_pair;
    execution_pair.program = *program;
    execution_pair.kernel = kernel;

    execution_pair.kernel_arg_count = 0;
    execution_pair.kernel_arg_prim_count = 0;
    execution_pair.kernel_arg_mem_obj_count = 0;
    execution_pair.kernel_arg_mem_objs = NULL;

    (*env)->xpair_count++;
    (*env)->xpairs = (cl_xpair_t*)realloc(
        (*env)->xpairs, (*env)->xpair_count * sizeof(cl_xpair_t));
    (*env)->xpairs[(*env)->xpair_count - 1] = execution_pair;

    printf("created execution pair, count is now %d\n", (*env)->xpair_count);

    return &((*env)->xpairs[(*env)->xpair_count - 1]);
}

cl_xpair_t* cl_add_kernel_arg_mem_obj(cl_env_t** env,
                                      cl_xpair_t* execution_pair,
                                      cl_uint position,
                                      size_t size,
                                      cl_mem mem_obj)
{
    assert(*env != NULL);
    assert(execution_pair != NULL);

    execution_pair->kernel_arg_count++;
    execution_pair->kernel_arg_mem_obj_count++;
    execution_pair->kernel_arg_mem_objs = (cl_mem*)realloc(
        execution_pair->kernel_arg_mem_objs,
        execution_pair->kernel_arg_mem_obj_count * sizeof(cl_mem));
    execution_pair
        ->kernel_arg_mem_objs[execution_pair->kernel_arg_mem_obj_count - 1] =
        mem_obj;

    CL_RET = clSetKernelArg(
        execution_pair->kernel,
        position,
        size,
        (const void*)&(execution_pair->kernel_arg_mem_objs
                           [execution_pair->kernel_arg_mem_obj_count - 1]));
    CL_CHECK_ERR(CL_RET);

    printf("set mem obj kernel arg, count is now %d\n",
           execution_pair->kernel_arg_count);

    return execution_pair;
}

cl_xpair_t* cl_add_kernel_arg_prim(cl_env_t** env,
                                   cl_xpair_t* execution_pair,
                                   cl_uint position,
                                   size_t size,
                                   const void* _ptr)
{
    assert(*env != NULL);
    assert(execution_pair != NULL);

    CL_RET = clSetKernelArg(execution_pair->kernel, position, size, _ptr);
    CL_CHECK_ERR(CL_RET);

    execution_pair->kernel_arg_count++;
    execution_pair->kernel_arg_prim_count++;

    printf("set primitive kernel arg, count is now %d\n",
           execution_pair->kernel_arg_prim_count);

    return execution_pair;
}

cl_xpair_t* cl_enqueue_kernel(cl_env_t** env,
                              cl_xpair_t* execution_pair,
                              cl_uint work_dimensions,
                              const size_t* _global_work_size,
                              const size_t* _local_work_size,
                              cl_event* event)
{
    assert(*env != NULL);
    assert(execution_pair != NULL);
    assert(_global_work_size != NULL);
    assert(_local_work_size != NULL);

    printf("enqueuing kernel with %d arguments\n",
           execution_pair->kernel_arg_count);

    CL_RET = clEnqueueNDRangeKernel((*env)->command_queue,
                                    execution_pair->kernel,
                                    work_dimensions,
                                    NULL,
                                    _global_work_size,
                                    _local_work_size,
                                    0,
                                    NULL,
                                    event);

    CL_CHECK_ERR(CL_RET);

    if (event != NULL)
    {
        CL_RET = clWaitForEvents(1, event);
        CL_CHECK_ERR(CL_RET);
    }

    return execution_pair;
}

void* cl_read_buffer(cl_env_t** env,
                     cl_mem* source_mem_obj,
                     cl_bool blocking_read,
                     size_t size,
                     void* ptr)
{
    CL_RET = clEnqueueReadBuffer((*env)->command_queue,
                                 *source_mem_obj,
                                 blocking_read,
                                 0,
                                 size,
                                 ptr,
                                 0,
                                 NULL,
                                 NULL);
    CL_CHECK_ERR(CL_RET);

    return ptr;
}

void cl_free(cl_env_t** env)
{
    assert(*env != NULL);

    CL_RET = clFlush((*env)->command_queue);
    CL_CHECK_ERR(CL_RET);
    CL_RET = clFinish((*env)->command_queue);
    CL_CHECK_ERR(CL_RET);

    for (int i = 0; i < (*env)->xpair_count; i++)
    {
        cl_xpair_t* xpair = &(*env)->xpairs[i];
        for (int j = 0; j < xpair->kernel_arg_mem_obj_count; j++)
        {
            CL_RET = clReleaseMemObject(xpair->kernel_arg_mem_objs[j]);
            CL_CHECK_ERR(CL_RET);
        }
        CL_RET = clReleaseKernel(xpair->kernel);
        CL_CHECK_ERR(CL_RET);
    }

    if ((*env)->xpair_count > 0) free((*env)->xpairs);

    for (int i = 0; i < (*env)->program_count; i++)
    {
        CL_RET = clReleaseProgram((*env)->programs[i]);
        CL_CHECK_ERR(CL_RET);
    }

    if ((*env)->program_count > 0) free((*env)->programs);

    CL_RET = clReleaseCommandQueue((*env)->command_queue);
    CL_CHECK_ERR(CL_RET);
    CL_RET = clReleaseContext((*env)->context);

    free(*env);
}

const char* cl_error_string(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:
        return "Success";
    case CL_DEVICE_NOT_FOUND:
        return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:
        return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
        return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
        return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
        return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
        return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
        return "Program build failure";
    case CL_MAP_FAILURE:
        return "Map failure";
    case CL_INVALID_VALUE:
        return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
        return "Invalid device type";
    case CL_INVALID_PLATFORM:
        return "Invalid platform";
    case CL_INVALID_DEVICE:
        return "Invalid device";
    case CL_INVALID_CONTEXT:
        return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
        return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
        return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
        return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
        return "Invalid image size";
    case CL_INVALID_SAMPLER:
        return "Invalid sampler";
    case CL_INVALID_BINARY:
        return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
        return "Invalid build options";
    case CL_INVALID_PROGRAM:
        return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
        return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
        return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
        return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
        return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
        return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
        return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
        return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
        return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
        return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "Invalid event wait list";
    case CL_INVALID_EVENT:
        return "Invalid event";
    case CL_INVALID_OPERATION:
        return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
        return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:
        return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
        return "Invalid mip-map level";
    default:
        return "Unknown error";
    }
}
