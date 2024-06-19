#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void convoluteGPU(float *dData, float *hData, int height, int width, float *mask, int masksize) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int S = (masksize - 1) / 2;
    float sum = 0;
    int pixPos = row * width + col;

    dData[pixPos] = 0.0;

    if (row < height && col < width) {
        for (int maskrow = -S; maskrow <= S; maskrow++) {
            for (int maskcol = -S; maskcol <= S; maskcol++) {
                int pixP = (row + maskrow) * width + (col + maskcol);
                int maskP = (maskrow + S) * masksize + (maskcol + S);

                if (pixP < height * width && pixP > 0 && maskP < masksize * masksize) {
                    sum += mask[maskP] * hData[pixP];
                }
            }
        }

        dData[pixPos] = sum;

        if (dData[pixPos] < 0) {
            dData[pixPos] = 0;
        } else if (dData[pixPos] > 1) {
            dData[pixPos] = 1;
        }
    }
}

int main() {
    const int height = 512;   // Adjust the height based on your data
    const int width = 512;    // Adjust the width based on your data
    const int masksize = 3;   // Adjust the mask size based on your requirement

    float *dData, *hData, *mask;

    // Allocate host memory
    hData = (float *)malloc(height * width * sizeof(float));
    dData = (float *)malloc(height * width * sizeof(float));
    mask = (float *)malloc(masksize * masksize * sizeof(float));

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < height * width; ++i) {
        hData[i] = static_cast<float>(i);
    }

    // Initialize mask (you may need to modify this based on your use case)
    for (int i = 0; i < masksize * masksize; ++i) {
        mask[i] = 1.0;
    }

    // Allocate device memory
    float *d_hData, *d_dData, *d_mask;
    cudaMalloc((void **)&d_hData, height * width * sizeof(float));
    cudaMalloc((void **)&d_dData, height * width * sizeof(float));
    cudaMalloc((void **)&d_mask, masksize * masksize * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_hData, hData, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, masksize * masksize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Launch the kernel
    convoluteGPU<<<grid_size, block_size>>>(d_dData, d_hData, height, width, d_mask, masksize);

    // Copy data from device to host
    cudaMemcpy(dData, d_dData, height * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_hData);
    cudaFree(d_dData);
    cudaFree(d_mask);
    free(hData);
    free(dData);
    free(mask);

    return 0;
}
 
