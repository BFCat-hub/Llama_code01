#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void transferMBR3(double* xy_in, long long* a_out, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        a_out[tid] = static_cast<long long>(xy_in[tid] * 10000000);
    }
}

int main() {
    int size = 1000;

    double* h_xy_in = (double*)malloc(size * sizeof(double));
    long long* h_a_out = (long long*)malloc(size * sizeof(long long));

    for (int i = 0; i < size; ++i) {
        h_xy_in[i] = static_cast<double>(i);
    }

    double* d_xy_in;
    long long* d_a_out;
    cudaMalloc((void**)&d_xy_in, size * sizeof(double));
    cudaMalloc((void**)&d_a_out, size * sizeof(long long));

    cudaMemcpy(d_xy_in, h_xy_in, size * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    transferMBR3<<<gridSize, blockSize>>>(d_xy_in, d_a_out, size);

    cudaMemcpy(h_a_out, d_a_out, size * sizeof(long long), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%ld ", h_a_out[i]);
    }

    free(h_xy_in);
    free(h_a_out);
    cudaFree(d_xy_in);
    cudaFree(d_a_out);

    return 0;
}