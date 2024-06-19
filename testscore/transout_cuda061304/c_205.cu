#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void vecAdd(float* in1, float* in2, float* out, size_t size) {
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < size) {
        out[gid] = in1[gid] + in2[gid];
    }
}

int main() {
    size_t size = 1000;

    float* h_in1 = (float*)malloc(size * sizeof(float));
    float* h_in2 = (float*)malloc(size * sizeof(float));
    float* h_out = (float*)malloc(size * sizeof(float));

    for (size_t i = 0; i < size; ++i) {
        h_in1[i] = static_cast<float>(i);
        h_in2[i] = static_cast<float>(2 * i);
    }

    float* d_in1;
    float* d_in2;
    float* d_out;
    cudaMalloc((void**)&d_in1, size * sizeof(float));
    cudaMalloc((void**)&d_in2, size * sizeof(float));
    cudaMalloc((void**)&d_out, size * sizeof(float));

    cudaMemcpy(d_in1, h_in1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1);

    vecAdd<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, size);

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 10; ++i) {
        printf("%f ", h_out[i]);
    }

    free(h_in1);
    free(h_in2);
    free(h_out);
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);

    return 0;
}