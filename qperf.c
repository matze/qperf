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

static gchar *ocl_read_program(const gchar *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
        return NULL;

    fseek(fp, 0, SEEK_END);
    const size_t length = ftell(fp);
    rewind(fp);

    gchar *buffer = (gchar *) g_malloc0(length);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }

    size_t buffer_length = fread(buffer, 1, length, fp);
    fclose(fp);
    if (buffer_length != length) {
        g_free(buffer);
        return NULL;
    }
    return buffer;
}

cl_program ocl_get_program(opencl_desc *ocl, const gchar *filename, const gchar *options)
{
    gchar *buffer = ocl_read_program(filename);
    if (buffer == NULL) 
        return FALSE;

    int errcode = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ocl->context, 1, (const char **) &buffer, NULL, &errcode);

    if (errcode != CL_SUCCESS) {
        g_free(buffer);
        return NULL;
    }

    errcode = clBuildProgram(program, ocl->num_devices, ocl->devices, options, NULL, NULL);

    if (errcode != CL_SUCCESS) {
        const int LOG_SIZE = 4096;
        gchar* log = (gchar *) g_malloc0(LOG_SIZE * sizeof(char));
        CHECK_ERROR(clGetProgramBuildInfo(program, ocl->devices[0], CL_PROGRAM_BUILD_LOG, LOG_SIZE, (void*) log, NULL));
        g_print("\n=== Build log for %s===%s\n\n", filename, log);
        g_free(log);
        g_free(buffer);
        return NULL;
    }

    g_free(buffer);
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

static test_environment *env_new(opencl_desc *ocl, int width, int height, int num_images)
{
    cl_int errcode = CL_SUCCESS;
    test_environment *env = g_malloc0(sizeof(test_environment));

    env->program = ocl_get_program(ocl, "simple.cl", "");
    if (env->program == NULL) {
        g_warning("Could not open simple.cl");
        ocl_free(ocl);
        return NULL;
    }

    /* Create kernel for each device */
    env->kernel = clCreateKernel(env->program, "test", &errcode);
    CHECK_ERROR(errcode);

    /* Generate four data images */
    size_t image_size = width * height * sizeof(float);
    env->num_images = num_images;
    env->host_data = (float **) g_malloc0(num_images * sizeof(float *));
    env->dev_data_in = (cl_mem *) g_malloc0(num_images * sizeof(cl_mem));
    env->dev_data_out = (cl_mem *) g_malloc0(num_images * sizeof(cl_mem));

    for (int i = 0; i < num_images; i++) {
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

int profile_queue(gboolean use_queue_profiling, int width, int height, int num_images)
{
    cl_int errcode = CL_SUCCESS;
    opencl_desc *ocl = ocl_new(use_queue_profiling);
    test_environment *env = env_new(ocl, width, height, num_images);
    if (env == NULL)
        return 1;

    cl_event events[num_images];
    cl_event read_events[num_images];
    size_t global_work_size[2] = { width, height };

    /* Measure single GPU case */
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
    g_timer_stop(timer);
    g_print("\t%f", g_timer_elapsed(timer, NULL));
    g_timer_destroy(timer);
    
    for (int i = 0; i < env->num_images; i++)
        CHECK_ERROR(clReleaseEvent(events[i]));

    env_free(env);
    ocl_free(ocl);
    return 0;
}

int main(int argc, char const* argv[])
{
    g_print("# width height num_images no_profiling profiling\n");
    for (int i = 32; i < 256; i += 32) {
        for (int j = 32; j < 256; j += 32) {
            for (int k = 16; k < 64; k+= 8) {
                g_print("%i\t%i\t%i", i, j, k);
                profile_queue(FALSE, i, j, k);
                profile_queue(TRUE, i, j, k);
                g_print("\n");
            }
        }
    }

    return 0;
}

