#include <device_launch_parameters.h>
#include <stdio.h>

// 双数组标量乘法（CPU 版本）
void doubleArrayScalarMultiply_cpu(double* d_in, double* d_out, int length, double scalar) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] * scalar;
    }
}

// 双数组标量乘法（GPU 版本）
__global__ void doubleArrayScalarMultiply_gpu(double* d_in, double* d_out, int length, double scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < length) {
        d_out[tid] = d_in[tid] * scalar;
    }
}

int main() {
    // 数组长度
    int length = 1000;

    // 在 CPU 上进行双数组标量乘法
    double scalar = 2.0;
    double* h_d_in = (double*)malloc(length * sizeof(double));
    double* h_d_out = (double*)malloc(length * sizeof(double));

    for (int i = 0; i < length; ++i) {
        h_d_in[i] = static_cast<double>(i);
    }

    doubleArrayScalarMultiply_cpu(h_d_in, h_d_out, length, scalar);

    // 在 GPU 上进行双数组标量乘法
    double* d_d_in;
    double* d_d_out;
    cudaMalloc((void**)&d_d_in, length * sizeof(double));
    cudaMalloc((void**)&d_d_out, length * sizeof(double));

    cudaMemcpy(d_d_in, h_d_in, length * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    doubleArrayScalarMultiply_gpu<<<gridSize, blockSize>>>(d_d_in, d_d_out, length, scalar);

    cudaMemcpy(h_d_out, d_d_out, length * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_d_out[i]);
    }

    free(h_d_in);
    free(h_d_out);
    cudaFree(d_d_in);
    cudaFree(d_d_out);

    return 0;
}