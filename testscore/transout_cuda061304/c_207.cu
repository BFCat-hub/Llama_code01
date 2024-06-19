#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void add_matrix(double* a, double* b, double* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = 0; i < N; i += stride) {
        c[i + idx] = a[i + idx] + b[i + idx];
    }
}

int main() {
    
    int N = 1000;

    
    double* h_a = (double*)malloc(N * sizeof(double));
    double* h_b = (double*)malloc(N * sizeof(double));
    double* h_c = (double*)malloc(N * sizeof(double));

    
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<double>(i);
        h_b[i] = static_cast<double>(2 * i);
    }

    
    double* d_a;
    double* d_b;
    double* d_c;
    cudaMalloc((void**)&d_a, N * sizeof(double));
    cudaMalloc((void**)&d_b, N * sizeof(double));
    cudaMalloc((void**)&d_c, N * sizeof(double));

    
    cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    
    add_matrix<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    
    cudaMemcpy(h_c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

    
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