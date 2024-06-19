#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void Mul_half(float* src, float* dst) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    dst[index] = src[index] * 0.5;
}

int main() {
    int arraySize = 1000;

    float* h_src = (float*)malloc(arraySize * sizeof(float));
    float* h_dst = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_src[i] = static_cast<float>(i);
    }

    float* d_src;
    float* d_dst;
    cudaMalloc((void**)&d_src, arraySize * sizeof(float));
    cudaMalloc((void**)&d_dst, arraySize * sizeof(float));

    cudaMemcpy(d_src, h_src, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    Mul_half<<<gridSize, blockSize>>>(d_src, d_dst);

    cudaMemcpy(h_dst, d_dst, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_dst[i]);
    }

    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}