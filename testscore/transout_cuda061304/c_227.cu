#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void vectorAdd(float* arrayA, float* arrayB, float* output, size_t numElements) {
    size_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadIndex < numElements) {
        output[threadIndex] = arrayA[threadIndex] + arrayB[threadIndex];
    }
}

int main() {
    
    size_t numElements = 1000;

    
    float* h_arrayA = (float*)malloc(numElements * sizeof(float));
    float* h_arrayB = (float*)malloc(numElements * sizeof(float));
    float* h_output = (float*)malloc(numElements * sizeof(float));

    
    for (size_t i = 0; i < numElements; ++i) {
        h_arrayA[i] = static_cast<float>(i);
        h_arrayB[i] = static_cast<float>(2 * i);
    }

    
    float* d_arrayA;
    float* d_arrayB;
    float* d_output;
    cudaMalloc((void**)&d_arrayA, numElements * sizeof(float));
    cudaMalloc((void**)&d_arrayB, numElements * sizeof(float));
    cudaMalloc((void**)&d_output, numElements * sizeof(float));

    
    cudaMemcpy(d_arrayA, h_arrayA, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrayB, h_arrayB, numElements * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x, 1);

    
    vectorAdd<<<gridSize, blockSize>>>(d_arrayA, d_arrayB, d_output, numElements);

    
    cudaMemcpy(h_output, d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (size_t i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }

    
    free(h_arrayA);
    free(h_arrayB);
    free(h_output);
    cudaFree(d_arrayA);
    cudaFree(d_arrayB);
    cudaFree(d_output);

    return 0;
}