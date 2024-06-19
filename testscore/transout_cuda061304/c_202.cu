#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void gpu_add(float* c, float* a, float* b, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < n) {
        c[j] = a[j] + b[j];
    }
}

int main() {
    
    int array_size = 1000;

    
    float* h_a = (float*)malloc(array_size * sizeof(float));
    float* h_b = (float*)malloc(array_size * sizeof(float));
    float* h_c = (float*)malloc(array_size * sizeof(float));

    
    for (int i = 0; i < array_size; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, array_size * sizeof(float));
    cudaMalloc((void**)&d_b, array_size * sizeof(float));
    cudaMalloc((void**)&d_c, array_size * sizeof(float));

    
    cudaMemcpy(d_a, h_a, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, array_size * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (array_size + blockSize - 1) / blockSize;

    
    gpu_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, array_size);

    
    cudaMemcpy(h_c, d_c, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    
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