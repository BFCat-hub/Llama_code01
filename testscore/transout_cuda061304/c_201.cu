#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void addKernel(float* input, float value, float* output) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    output[gid] = input[gid] + value;
}

int main() {
    
    int arraySize = 1000;

    
    float* h_input = (float*)malloc(arraySize * sizeof(float));
    float* h_value = (float*)malloc(arraySize * sizeof(float));
    float* h_output = (float*)malloc(arraySize * sizeof(float));

    
    for (int i = 0; i < arraySize; ++i) {
        h_input[i] = static_cast<float>(i);
        h_value[i] = static_cast<float>(2 * i);
    }

    
    float* d_input;
    float* d_value;
    float* d_output;
    cudaMalloc((void**)&d_input, arraySize * sizeof(float));
    cudaMalloc((void**)&d_value, arraySize * sizeof(float));
    cudaMalloc((void**)&d_output, arraySize * sizeof(float));

    
    cudaMemcpy(d_input, h_input, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    
    addKernel<<<gridSize, blockSize>>>(d_input, d_value[0], d_output);

    
    cudaMemcpy(h_output, d_output, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }

    
    free(h_input);
    free(h_value);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_value);
    cudaFree(d_output);

    return 0;
}