#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void test1(float* input, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    if (input[tid * 4] != 0) {
        input[tid * 4] = 0;
    }
}

int main() {
    int dims = 1000;

    float* h_input = (float*)malloc(dims * 4 * sizeof(float));

    float* d_input;
    cudaMalloc((void**)&d_input, dims * 4 * sizeof(float));

    cudaMemcpy(d_input, h_input, dims * 4 * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (dims + blockSize - 1) / blockSize;

    test1<<<gridSize, blockSize>>>(d_input, dims);

    cudaMemcpy(h_input, d_input, dims * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; ++i) {
        printf("h_input[%d]: %f\n", i, h_input[i]);
    }

    free(h_input);
    cudaFree(d_input);

    return 0;
}