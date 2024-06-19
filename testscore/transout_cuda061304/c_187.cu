#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void incrementArrayOnGPU(float* a, int N) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < N; i++) {
        int address = i * num_threads + thread_index;

        if (address < N * num_threads) {
            a[address] = a[address] + 1.0f;
        }
    }
}

int main() {
    int array_size = 1000;

    float* h_a = (float*)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    float* d_a;
    cudaMalloc((void**)&d_a, array_size * sizeof(float));

    cudaMemcpy(d_a, h_a, array_size * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (array_size + block_size - 1) / block_size;

    incrementArrayOnGPU<<<grid_size, block_size>>>(d_a, array_size);

    cudaMemcpy(h_a, d_a, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_a[i]);
    }

    free(h_a);
    cudaFree(d_a);

    return 0;
}