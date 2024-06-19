#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void subtract_matrix(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        c[idx] = a[idx] - b[idx];
    }
}

int main() {
    
    int N = 1000;

    
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));
    float* h_c = (float*)malloc(N * sizeof(float));

    
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    
    subtract_matrix<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    
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