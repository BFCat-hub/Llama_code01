#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void allExp2InplaceKernel(double* arr, int n) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < n; i += num_threads) {
        int index = i + thread_index;

        if (index < n) {
            arr[index] = arr[index] * 9.0;
        }
    }
}

int main() {
    int n = 1000;

    double* h_arr = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        h_arr[i] = static_cast<double>(i);
    }

    double* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(double));

    cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    allExp2InplaceKernel<<<gridSize, blockSize>>>(d_arr, n);

    cudaMemcpy(h_arr, d_arr, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_arr[i]);
    }

    free(h_arr);
    cudaFree(d_arr);

    return 0;
}