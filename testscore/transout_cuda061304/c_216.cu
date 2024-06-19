#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void doubleArrayElementwiseSquareKernel(double* d_in, double* d_out, int length) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < length) {
        d_out[tid] = d_in[tid] * d_in[tid];
    }
}

int main() {
    int arraySize = 1000;

    double* h_d_in = (double*)malloc(arraySize * sizeof(double));
    double* h_d_out = (double*)malloc(arraySize * sizeof(double));

    for (int i = 0; i < arraySize; ++i) {
        h_d_in[i] = static_cast<double>(i);
    }

    double* d_d_in;
    double* d_d_out;
    cudaMalloc((void**)&d_d_in, arraySize * sizeof(double));
    cudaMalloc((void**)&d_d_out, arraySize * sizeof(double));

    cudaMemcpy(d_d_in, h_d_in, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    doubleArrayElementwiseSquareKernel<<<gridSize, blockSize>>>(d_d_in, d_d_out, arraySize);

    cudaMemcpy(h_d_out, d_d_out, arraySize * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_d_out[i]);
    }

    free(h_d_in);
    free(h_d_out);
    cudaFree(d_d_in);
    cudaFree(d_d_out);

    return 0;
}