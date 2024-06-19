#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void saxpy(float* x, float* y, float alpha, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main() {
    
    int array_size = 1000;

    
    float* h_x = (float*)malloc(array_size * sizeof(float));
    float* h_y = (float*)malloc(array_size * sizeof(float));

    
    for (int i = 0; i < array_size; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(2 * i);
    }

    
    float* d_x;
    float* d_y;
    cudaMalloc((void**)&d_x, array_size * sizeof(float));
    cudaMalloc((void**)&d_y, array_size * sizeof(float));

    
    cudaMemcpy(d_x, h_x, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, array_size * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (array_size + blockSize - 1) / blockSize;

    
    saxpy<<<gridSize, blockSize>>>(d_x, d_y, 2.0f, array_size);

    
    cudaMemcpy(h_y, d_y, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_y[i]);
    }

    
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}