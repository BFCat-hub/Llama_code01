#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void Ring_kernel(float *A, float *BP, int *corrAB, float *M, int ring, int c, int h, int w) {
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    int size = h * w;
    int ringSize = 2 * ring + 1;
    int ringPatch = ringSize * ringSize;

    if (id1 < size) {
        int y1 = id1 / w, x1 = id1 % w;
        int y2 = corrAB[2 * id1 + 1], x2 = corrAB[2 * id1 + 0];

        for (int dx = -ring; dx <= ring; dx++)
            for (int dy = -ring; dy <= ring; dy++) {
                int pIdx = (dy + ring) * ringSize + (dx + ring);
                int _x2 = x2 + dx, _y2 = y2 + dy;

                if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h) {
                    for (int dc = 0; dc < c; dc++) {
                        M[(dc * size + y1 * w) * ringPatch + pIdx * w + x1] = BP[dc * size + _y2 * w + _x2];
                    }
                }
            }
    }
}

int main() {
    const int h = 512; // Adjust the height based on your data
    const int w = 512; // Adjust the width based on your data
    const int c = 3;   // Adjust the number of channels based on your data
    const int ring = 2; // Adjust the ring size based on your requirement

    float *A, *BP, *M;
    int *corrAB;

    // Allocate host memory
    A = (float *)malloc(h * w * sizeof(float));
    BP = (float *)malloc(c * h * w * sizeof(float));
    M = (float *)malloc(c * h * w * (2 * ring + 1) * (2 * ring + 1) * sizeof(float));
    corrAB = (int *)malloc(2 * h * w * sizeof(int));

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < h * w; ++i) {
        A[i] = static_cast<float>(i);
        corrAB[2 * i] = i % w;
        corrAB[2 * i + 1] = i / w;
    }

    for (int i = 0; i < c * h * w; ++i) {
        BP[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_A, *d_BP, *d_M;
    int *d_corrAB;
    cudaMalloc((void **)&d_A, h * w * sizeof(float));
    cudaMalloc((void **)&d_BP, c * h * w * sizeof(float));
    cudaMalloc((void **)&d_M, c * h * w * (2 * ring + 1) * (2 * ring + 1) * sizeof(float));
    cudaMalloc((void **)&d_corrAB, 2 * h * w * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, A, h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BP, BP, c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_corrAB, corrAB, 2 * h * w * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((h * w + block_size.x - 1) / block_size.x);

    // Launch the kernel
    Ring_kernel<<<grid_size, block_size>>>(d_A, d_BP, d_corrAB, d_M, ring, c, h, w);

    // Copy data from device to host
    cudaMemcpy(M, d_M, c * h * w * (2 * ring + 1) * (2 * ring + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_BP);
    cudaFree(d_M);
    cudaFree(d_corrAB);
    free(A);
    free(BP);
    free(M);
    free(corrAB);

    return 0;
}
 
