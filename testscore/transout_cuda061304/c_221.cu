#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vectorDiv(float* A, float* B, float* C, int numElements) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < numElements) {
        C[gid] = A[gid] / B[gid];
    }
}

int main() {
    
    int numElements = 1000;

    
    float* h_A = (float*)malloc(numElements * sizeof(float));
    float* h_B = (float*)malloc(numElements * sizeof(float));
    float* h_C = (float*)malloc(numElements * sizeof(float));

    
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, numElements * sizeof(float));
    cudaMalloc((void**)&d_B, numElements * sizeof(float));
    cudaMalloc((void**)&d_C, numElements * sizeof(float));

    
    cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    
    vectorDiv<<<gridSize, blockSize>>>(d_A, d_B, d_C, numElements);

    
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    
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