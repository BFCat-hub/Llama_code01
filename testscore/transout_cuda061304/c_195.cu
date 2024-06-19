#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void sigmoid_kernel(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        output[tid] = 1.0 / (1.0 + expf(-input[tid]));
    }
}

int main() {
    int array_size = 1000;

    float* h_input = (float*)malloc(array_size * sizeof(float));
    float* h_output = (float*)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, array_size * sizeof(float));
    cudaMalloc((void**)&d_output, array_size * sizeof(float));

    cudaMemcpy(d_input, h_input, array_size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (array_size + blockSize - 1) / blockSize;

    sigmoid_kernel<<<gridSize, blockSize>>>(d_input, d_output, array_size);

    cudaMemcpy(h_output, d_output, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}