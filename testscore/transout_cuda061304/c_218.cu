#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void k_vec_divide(float* vec1, float* vec2, int max_size) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < max_size) {
        vec1[gid] = vec1[gid] / vec2[gid];
    }
}

int main() {
    int max_size = 1000;

    float* h_vec1 = (float*)malloc(max_size * sizeof(float));
    float* h_vec2 = (float*)malloc(max_size * sizeof(float));

    for (int i = 0; i < max_size; i++) {
        h_vec1[i] = static_cast<float>(i);
        h_vec2[i] = static_cast<float>(i * 2);
    }

    float* d_vec1;
    float* d_vec2;
    cudaMalloc((void**)&d_vec1, max_size * sizeof(float));
    cudaMalloc((void**)&d_vec2, max_size * sizeof(float));

    cudaMemcpy(d_vec1, h_vec1, max_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, max_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((max_size + blockSize.x - 1) / blockSize.x, 1);

    k_vec_divide<<<gridSize, blockSize>>>(d_vec1, d_vec2, max_size);

    cudaMemcpy(h_vec1, d_vec1, max_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_vec1[i]);
    }

    free(h_vec1);
    free(h_vec2);
    cudaFree(d_vec1);
    cudaFree(d_vec2);

    return 0;
}