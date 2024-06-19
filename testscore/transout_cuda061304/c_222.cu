#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void cudaSAXPY(int N, float alpha, float* X, float* Y) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] = alpha * X[i] + Y[i];
    }
}

int main() {
    int array_size = 1000;

    float* h_X = (float*)malloc(array_size * sizeof(float));
    float* h_Y = (float*)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        h_X[i] = static_cast<float>(i);
        h_Y[i] = static_cast<float>(2 * i);
    }

    float* d_X;
    float* d_Y;
    cudaMalloc((void**)&d_X, array_size * sizeof(float));
    cudaMalloc((void**)&d_Y, array_size * sizeof(float));

    cudaMemcpy(d_X, h_X, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, array_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((array_size + blockSize.x - 1) / blockSize.x, 1);

    cudaSAXPY<<<gridSize, blockSize>>>(array_size, 2.0f, d_X, d_Y);

    cudaMemcpy(h_Y, d_Y, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_Y[i]);
    }

    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}