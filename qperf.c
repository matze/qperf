/* qperf.c
 *
 * Copyright (C) 2011 Matthias Vogelgesang <matthias.vogelgesang@gmail.com>
 *
 * qperf is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * qperf is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Labyrinth; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

#include <CL/cl.h>
#include <glib-2.0/glib.h>
#include <stdio.h>

typedef struct {
    cl_context context;
    cl_uint num_devices;
    cl_device_id *devices;
    cl_command_queue *cmd_queues;

    GList *kernel_table;
    GHashTable *kernels;         /**< maps from kernel string to cl_kernel */
} opencl_desc;

typedef struct {
    cl_program program;
    cl_kernel kernel;
    int num_images;
    size_t image_size;
    float **host_data;
    cl_mem *dev_data_in;
    cl_mem *dev_data_out;
    guint width;
    guint height;
} test_environment;

static const gchar* opencl_error_msgs[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",

    /* next IDs start at 30! */
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE"
};

const gchar* opencl_map_error(int error)
{
    if (error >= -14)
        return opencl_error_msgs[-error];
    if (error <= -30)
        return opencl_error_msgs[-error-15];
    return NULL;
}

#define CHECK_ERROR(error) { \
    if ((error) != CL_SUCCESS) g_message("OpenCL error <%s:%i>: %s", __FILE__, __LINE__, opencl_map_error((error))); }

cl_program ocl_get_program(opencl_desc *ocl, const gchar *source, const gchar *options)
{
    int errcode = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ocl->context, 1, (const char **) &source, NULL, &errcode);

    if (errcode != CL_SUCCESS)
        return NULL;

    errcode = clBuildProgram(program, ocl->num_devices, ocl->devices, options, NULL, NULL);

    if (errcode != CL_SUCCESS) {
        const int LOG_SIZE = 4096;
        gchar* log = (gchar *) g_malloc0(LOG_SIZE * sizeof(char));
        CHECK_ERROR(clGetProgramBuildInfo(program, ocl->devices[0], CL_PROGRAM_BUILD_LOG, LOG_SIZE, (void*) log, NULL));
        g_print("\n=== Build log ===%s\n\n", log);
        g_free(log);
        return NULL;
    }

    return program;
}

opencl_desc *ocl_new(gboolean profile)
{
    opencl_desc *ocl = g_malloc0(sizeof(opencl_desc));

    cl_platform_id platform;
    int errcode = CL_SUCCESS;
    CHECK_ERROR(clGetPlatformIDs(1, &platform, NULL));

    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ocl->num_devices));
    ocl->devices = g_malloc0(ocl->num_devices * sizeof(cl_device_id));
    CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, ocl->num_devices, ocl->devices, NULL));

    ocl->context = clCreateContext(NULL, ocl->num_devices, ocl->devices, NULL, NULL, &errcode);
    CHECK_ERROR(errcode);

    ocl->cmd_queues = g_malloc0(ocl->num_devices * sizeof(cl_command_queue));
    cl_command_queue_properties queue_properties = profile ? CL_QUEUE_PROFILING_ENABLE : 0;

    const size_t len = 256;
    char device_name[len];
    for (int i = 0; i < ocl->num_devices; i++) {
        CHECK_ERROR(clGetDeviceInfo(ocl->devices[i], CL_DEVICE_NAME, len, device_name, NULL));
        ocl->cmd_queues[i] = clCreateCommandQueue(ocl->context, ocl->devices[i], queue_properties, &errcode);
        CHECK_ERROR(errcode);
    }
    return ocl;
}

static void ocl_free(opencl_desc *ocl)
{
    for (int i = 0; i < ocl->num_devices; i++)
        clReleaseCommandQueue(ocl->cmd_queues[i]);

    CHECK_ERROR(clReleaseContext(ocl->context));

    g_free(ocl->devices);
    g_free(ocl->cmd_queues);
    g_free(ocl);
}

static test_environment *env_new(opencl_desc *ocl, int width, int height)
{
    static const char *source = "\
__kernel void test(__global float *input, __global float *output)\
{ \
    const int idx = get_global_id(1) * get_global_size(0) + get_global_id(0); \
    output[idx] = input[idx] * 2.0f; \
}";

    cl_int errcode = CL_SUCCESS;
    test_environment *env = g_malloc0(sizeof(test_environment));

    env->program = ocl_get_program(ocl, source, "");
    if (env->program == NULL) {
        ocl_free(ocl);
        return NULL;
    }

    /* Create kernel for each device */
    env->kernel = clCreateKernel(env->program, "test", &errcode);
    CHECK_ERROR(errcode);

    /* Generate four data images */
    size_t image_size = width * height * sizeof(float);
    env->width = width;
    env->height = height;
    env->num_images = 8;
    env->host_data = (float **) g_malloc0(env->num_images * sizeof(float *));
    env->dev_data_in = (cl_mem *) g_malloc0(env->num_images * sizeof(cl_mem));
    env->dev_data_out = (cl_mem *) g_malloc0(env->num_images * sizeof(cl_mem));

    for (int i = 0; i < env->num_images; i++) {
        env->host_data[i] = (float *) g_malloc0(image_size);
        env->dev_data_in[i] = clCreateBuffer(ocl->context, 
                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image_size, env->host_data[i], &errcode);
        CHECK_ERROR(errcode);
        env->dev_data_out[i] = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, image_size, NULL, &errcode);
        CHECK_ERROR(errcode);
    }
    return env;
}

static void env_free(test_environment *env)
{
    for (int i = 0; i < env->num_images; i++) {
        g_free(env->host_data[i]);
        CHECK_ERROR(clReleaseMemObject(env->dev_data_in[i]));
        CHECK_ERROR(clReleaseMemObject(env->dev_data_out[i]));
    }

    clReleaseKernel(env->kernel);
    clReleaseProgram(env->program);
    g_free(env->host_data);
    g_free(env);
}

gdouble profile_queue(opencl_desc *ocl, test_environment *env)
{
    gdouble result;
    cl_event events[env->num_images];
    cl_event read_events[env->num_images];
    size_t global_work_size[2] = { env->width, env->height };
    GTimer *timer = g_timer_new();

    for (int i = 0; i < env->num_images; i++) {
        CHECK_ERROR(clSetKernelArg(env->kernel, 0, sizeof(cl_mem), (void *) &env->dev_data_in[i]))
        CHECK_ERROR(clSetKernelArg(env->kernel, 1, sizeof(cl_mem), (void *) &env->dev_data_out[i]));
        
        CHECK_ERROR(clEnqueueNDRangeKernel(ocl->cmd_queues[0], env->kernel,
                2, NULL, global_work_size, NULL,
                0, NULL, &events[i]));

        CHECK_ERROR(clEnqueueReadBuffer(ocl->cmd_queues[0], 
                env->dev_data_out[i], CL_FALSE, 0, env->image_size, env->host_data[i], 
                1, &events[i], &read_events[i]));
    }

    clWaitForEvents(env->num_images, read_events);
    result = g_timer_elapsed(timer, NULL);
    g_timer_destroy(timer);
    
    for (int i = 0; i < env->num_images; i++)
        CHECK_ERROR(clReleaseEvent(events[i]));

    return result;
}

static void run_benchmark(gboolean use_queue_profiling)
{
    opencl_desc *ocl = ocl_new(use_queue_profiling);

    for (int i = 256; i < 4096; i *= 2) {
        /* for (int j = i; j < 2048; j *= 2) { */
            gdouble total_time = 0.0;
            gdouble min_time = G_MAXDOUBLE;
            gdouble max_time = 0.0;
            test_environment *env = env_new(ocl, i, i);

            for (int k = 0; k < 16; k++) {
                gdouble time = profile_queue(ocl, env);
                total_time += time;
                min_time = MIN(min_time, time);
                max_time = MAX(max_time, time);
            }

            g_print("%i %i %i %f %f %f\n", (int) use_queue_profiling,
                    i, i, total_time / 16.0, min_time, max_time);

            env_free(env);
        /* } */
    }

    ocl_free(ocl);
}

int main(int argc, char const* argv[])
{
    g_print("# profiling? width height avg min max\n");
    run_benchmark(TRUE);
    run_benchmark(FALSE);

    return 0;
}

