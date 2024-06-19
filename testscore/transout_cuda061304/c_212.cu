#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < N; i += num_threads) {
        int index = i + thread_index;

        if (index < N) {
            C[index] = A[index] + B[index];
        }
    }
}

int main() {
    int array_size = 1000;

    float* h_array_A = (float*)malloc(array_size * sizeof(float));
    float* h_array_B = (float*)malloc(array_size * sizeof(float));
    float* h_array_C = (float*)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        h_array_A[i] = static_cast<float>(i);
        h_array_B[i] = static_cast<float>(2 * i);
    }

    float* d_array_A;
    float* d_array_B;
    float* d_array_C;
    cudaMalloc((void**)&d_array_A, array_size * sizeof(float));
    cudaMalloc((void**)&d_array_B, array_size * sizeof(float));
    cudaMalloc((void**)&d_array_C, array_size * sizeof(float));

    cudaMemcpy(d_array_A, h_array_A, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array_B, h_array_B, array_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((array_size + blockSize.x - 1) / blockSize.x, 1);

    VecAdd<<<gridSize, blockSize>>>(d_array_A, d_array_B, d_array_C, array_size);

    cudaMemcpy(h_array_C, d_array_C, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_array_C[i]);
    }

    free(h_array_A);
    free(h_array_B);
    free(h_array_C);
    cudaFree(d_array_A);
    cudaFree(d_array_B);
    cudaFree(d_array_C);

    return 0;
}