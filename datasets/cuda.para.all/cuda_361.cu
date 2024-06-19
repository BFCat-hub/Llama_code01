#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void nlf_right_forward(const int n, const float *filters, const int channel,
                                   const int height, const int width, const int wsize,
                                   float *top_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    int step = height * width;
    int base = index * step;
    int fbase = index / channel * wsize * step;

    for (int col = 0; col < width; col++) {
        for (int row = 0; row < height; row++) {
            float temp = 0;
            int r, c, shift;

            // Center
            r = row;
            c = col;
            shift = 0 * step + row * width + col;
            temp += top_data[base + r * width + c] * filters[fbase + shift];

            // Left
            r = row;
            c = col - 1;
            shift = 1 * step + row * width + col;
            if (c >= 0)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            // Top-left
            r = row - 1;
            c = col - 1;
            shift = 2 * step + row * width + col;
            if (c >= 0 && r >= 0)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            // Bottom-left
            r = row + 1;
            c = col - 1;
            shift = 3 * step + row * width + col;
            if (c >= 0 && r < height)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            // Top
            r = row - 1;
            c = col;
            shift = 4 * step + row * width + col;
            if (r >= 0)
                temp += top_data[base + r * width + c] * filters[fbase + shift];
            else
                temp += top_data[base + row * width + col] * filters[fbase + shift];

            top_data[base + row * width + col] = temp;
        }
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
    float *filters_host = (float *)malloc(n * channel * wsize * sizeof(float));
    float *top_data_host = (float *)malloc(n * height * width * sizeof(float));

    // Initialize input data (filters, top_data) on the host

    // Allocate memory on the device
    float *filters_device, *top_data_device;

    cudaMalloc((void **)&filters_device, n * channel * wsize * sizeof(float));
    cudaMalloc((void **)&top_data_device, n * height * width * sizeof(float));

    // Copy input data from host to device

    // Launch the CUDA kernel
    dim3 gridDim((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);

    nlf_right_forward<<<gridDim, blockDim>>>(n, filters_device, channel, height, width, wsize,
                                             top_data_device);

    // Copy the result back from device to host

    // Free allocated memory on both host and device

    free(filters_host);
    free(top_data_host);

    cudaFree(filters_device);
    cudaFree(top_data_device);

    return 0;
}
 
