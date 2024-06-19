#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void sumArraysKernel(float* A, float* B, float* C, const int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    const int arraySize = 1000;

    float* h_A = (float*)malloc(arraySize * sizeof(float));
    float* h_B = (float*)malloc(arraySize * sizeof(float));
    float* h_C = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, arraySize * sizeof(float));
    cudaMalloc((void**)&d_B, arraySize * sizeof(float));
    cudaMalloc((void**)&d_C, arraySize * sizeof(float));

    cudaMemcpy(d_A, h_A, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    sumArraysKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, arraySize);

    cudaMemcpy(h_C, d_C, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_C[i]);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}