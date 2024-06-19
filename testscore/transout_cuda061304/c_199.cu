#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void clearArray(unsigned char* arr, const unsigned int length) {
    unsigned int offset = threadIdx.x + blockDim.x * blockIdx.x;
    while (offset < length) {
        arr[offset] = 0;
        offset += blockDim.x * gridDim.x;
    }
}

int main() {
    const unsigned int length = 1000;

    unsigned char* h_arr = (unsigned char*)malloc(length * sizeof(unsigned char));

    unsigned char* d_arr;
    cudaMalloc((void**)&d_arr, length * sizeof(unsigned char));

    cudaMemcpy(d_arr, h_arr, length * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((length + blockSize.x - 1) / blockSize.x, 1);

    clearArray<<<gridSize, blockSize>>>(d_arr, length);

    cudaMemcpy(h_arr, d_arr, length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < 10; ++i) {
        printf("%u ", h_arr[i]);
    }

    free(h_arr);
    cudaFree(d_arr);

    return 0;
}