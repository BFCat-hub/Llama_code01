#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void binarize_kernel(float* input, int n, float* binary) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        binary[i] = (input[i] > 0) ? 1.0f : -1.0f;
    }
}

int main() {
    int n = 1000;

    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_binary = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input;
    float* d_binary;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_binary, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1);

    binarize_kernel<<<gridSize, blockSize>>>(d_input, n, d_binary);

    cudaMemcpy(h_binary, d_binary, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("h_binary[%d]: %f\n", i, h_binary[i]);
    }

    free(h_input);
    free(h_binary);
    cudaFree(d_input);
    cudaFree(d_binary);

    return 0;
}