#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void nlf_filter_right_backward(const int n, const float *bottom_data, const float *top_data,
                                           const float *temp_diff, const int channel,
                                           const int height, const int width, const int wsize,
                                           float *filters_diff) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    int step = height * width;
    int base = index / step * step * channel + index % step;
    int fbase = index / step * step * wsize + index % step;
    int row = index % step / width;
    int col = index % step % width;

    for (int i = 0; i < channel; i++) {
        filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (col - 1 >= 0)
            filters_diff[fbase + step] += temp_diff[base + i * step] * top_data[base - 1 + i * step];
        else
            filters_diff[fbase + step] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (col - 1 >= 0 && row - 1 >= 0)
            filters_diff[fbase + 2 * step] +=
                temp_diff[base + i * step] * top_data[base - width - 1 + i * step];
        else
            filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (col - 1 >= 0 && row + 1 < height)
            filters_diff[fbase + 3 * step] +=
                temp_diff[base + i * step] * top_data[base + width - 1 + i * step];
        else
            filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

        if (row - 1 >= 0)
            filters_diff[fbase + 4 * step] +=
                temp_diff[base + i * step] * top_data[base - width + i * step];
        else
            filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];
    }
}

int main() {
    // Example usage
    int n = 1000;
    int channel = 3;
    int height = 32;
    int width = 32;
    int wsize = 5;

    // Allocate memory on the host
    float *bottom_data_host = (float *)malloc(n * channel * height * width * sizeof(float));
    float *top_data_host = (float *)malloc(n * channel * height * width * sizeof(float));
    float *temp_diff_host = (float *)malloc(n * channel * height * width * sizeof(float));
    float *filters_diff_host = (float *)malloc(n * channel * wsize * sizeof(float));

    // Initialize input data (bottom_data, top_data, temp_diff) on the host

    // Allocate memory on the device
    float *bottom_data_device, *top_data_device, *temp_diff_device, *filters_diff_device;

    cudaMalloc((void **)&bottom_data_device, n * channel * height * width * sizeof(float));
    cudaMalloc((void **)&top_data_device, n * channel * height * width * sizeof(float));
    cudaMalloc((void **)&temp_diff_device, n * channel * height * width * sizeof(float));
    cudaMalloc((void **)&filters_diff_device, n * channel * wsize * sizeof(float));

    // Copy input data from host to device

    // Launch the CUDA kernel
    dim3 gridDim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);

    nlf_filter_right_backward<<<gridDim, blockDim>>>(n, bottom_data_device, top_data_device,
                                                     temp_diff_device, channel, height, width,
                                                     wsize, filters_diff_device);

    // Copy the result back from device to host

    // Free allocated memory on both host and device

    free(bottom_data_host);
    free(top_data_host);
    free(temp_diff_host);
    free(filters_diff_host);

    cudaFree(bottom_data_device);
    cudaFree(top_data_device);
    cudaFree(temp_diff_device);
    cudaFree(filters_diff_device);

    return 0;
}
 
