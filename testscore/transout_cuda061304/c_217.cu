#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void sumArraysOnGPU(int* A, int* B, int* C, const int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    
    const int N = 1000;

    
    int* h_A = (int*)malloc(N * sizeof(int));
    int* h_B = (int*)malloc(N * sizeof(int));
    int* h_C = (int*)malloc(N * sizeof(int));

    
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    
    int* d_A;
    int* d_B;
    int* d_C;
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    
    sumArraysOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    
    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_C[i]);
    }

    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}