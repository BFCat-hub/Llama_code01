#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void const_kernel(int N, float ALPHA, float* X, int INCX) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
        X[i * INCX] = ALPHA;
}

int main() {
    int N = 1000;
    int INCX = 1;
    float ALPHA = 2.0;

    float* h_X = (float*)malloc(N * sizeof(float));

    float* d_X;
    cudaMalloc((void**)&d_X, N * sizeof(float));

    dim3 gridSize((N + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    const_kernel<<<gridSize, blockSize>>>(N, ALPHA, d_X, INCX);

    cudaMemcpy(h_X, d_X, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_X[i]);
    }

    free(h_X);
    cudaFree(d_X);

    return 0;
}