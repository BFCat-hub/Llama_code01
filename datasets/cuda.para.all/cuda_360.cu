#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void calculateOuterSumsNew(float *innerSums, float *L, int uLength) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if (u >= uLength)
        return;

    float real, imag, u_sum = 0.0f;

    for (int i = 0; i < 8; i++) {
        int realIdx = 2 * (u + i * 64);
        int imagIdx = realIdx + 1;

        real = innerSums[realIdx];
        imag = innerSums[imagIdx];

        u_sum += (real * real) + (imag * imag);
    }

    L[u] = u_sum;
}

int main() {
    // Example usage
    int uLength = 1000;

    // Allocate memory on the host
    float *innerSums_host = (float *)malloc(2 * uLength * sizeof(float));
    float *L_host = (float *)malloc(uLength * sizeof(float));

    // Initialize input data (innerSums) on the host

    // Allocate memory on the device
    float *innerSums_device, *L_device;

    cudaMalloc((void **)&innerSums_device, 2 * uLength * sizeof(float));
    cudaMalloc((void **)&L_device, uLength * sizeof(float));

    // Copy input data from host to device

    // Launch the CUDA kernel
    dim3 gridDim((uLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);

    calculateOuterSumsNew<<<gridDim, blockDim>>>(innerSums_device, L_device, uLength);

    // Copy the result back from device to host

    // Free allocated memory on both host and device

    free(innerSums_host);
    free(L_host);

    cudaFree(innerSums_device);
    cudaFree(L_device);

    return 0;
}
 
