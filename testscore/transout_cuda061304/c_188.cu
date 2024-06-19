#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void allMulInplace(double* arr, double alpha, size_t n) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        arr[i] *= alpha;
    }
}

int main() {
    size_t n = 1000;

    double* h_arr = (double*)malloc(n * sizeof(double));

    for (size_t i = 0; i < n; ++i) {
        h_arr[i] = static_cast<double>(i);
    }

    double* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(double));

    cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    allMulInplace<<<gridSize, blockSize>>>(d_arr, 2.0, n);

    cudaMemcpy(h_arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10; ++i) {
        printf("%f ", h_arr[i]);
    }

    free(h_arr);
    cudaFree(d_arr);

    return 0;
}