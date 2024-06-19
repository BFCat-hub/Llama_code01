#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void saxpi(int n, float a, float* x, float* y) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    printf("SAXPI_CUDA sample\n");

    
    int n = 1000;
    float a = 0.1;

    
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(n * sizeof(float));

    
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(i * 2);
    }

    
    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1);

    
    saxpi<<<gridSize, blockSize>>>(n, a, d_x, d_y);

    
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("y[%d]: %f\n", i, h_y[i]);
    }

    
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}