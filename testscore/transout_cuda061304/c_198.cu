#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void allLog2(double* arr, double* buf, int n) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < n; i += num_threads) {
        int index = i + thread_index;

        if (index < n) {
            buf[index] = arr[index] / 2.0;
        }
    }
}

int main() {
    int n = 1000;

    double* h_arr = (double*)malloc(n * sizeof(double));
    double* h_buf = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; ++i) {
        h_arr[i] = static_cast<double>(i);
    }

    double* d_arr;
    double* d_buf;
    cudaMalloc((void**)&d_arr, n * sizeof(double));
    cudaMalloc((void**)&d_buf, n * sizeof(double));

    cudaMemcpy(d_arr, h_arr, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    allLog2<<<gridSize, blockSize>>>(d_arr, d_buf, n);

    cudaMemcpy(h_buf, d_buf, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_buf[i]);
    }

    free(h_arr);
    free(h_buf);
    cudaFree(d_arr);
    cudaFree(d_buf);

    return 0;
}