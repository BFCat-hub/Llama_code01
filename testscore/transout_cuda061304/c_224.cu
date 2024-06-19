#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void vectorAdd(float* a, float* b, float* c, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    
    int arraySize = 1000;

    
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(float));
    cudaMalloc((void**)&d_b, arraySize * sizeof(float));
    cudaMalloc((void**)&d_c, arraySize * sizeof(float));

    
    cudaMemcpy(d_a, h_a, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, arraySize);

    
    cudaMemcpy(h_c, d_c, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_c[i]);
    }

    
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}