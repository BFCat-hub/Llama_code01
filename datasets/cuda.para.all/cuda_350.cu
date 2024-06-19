#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void convolution_kernel_v1(float *device_outputMatrix, float *device_inputMatrix, float *device_filter, int imageRows, int imageColumns, int filterSize) {
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    float convolvedValue = 0.f;

    for (int eachFilterRow = -filterSize / 2; eachFilterRow <= filterSize / 2; ++eachFilterRow) {
        for (int eachFilterColumn = -filterSize / 2; eachFilterColumn <= filterSize / 2; ++eachFilterColumn) {
            int imageRow = index_y + eachFilterRow;
            int imageColumn = index_x + eachFilterColumn;

            float pixelValue = (imageRow >= 0 && imageRow < imageRows && imageColumn >= 0 && imageColumn < imageColumns)
                                   ? device_inputMatrix[imageRow * imageColumns + imageColumn]
                                   : 0.f;

            float filterValue = device_filter[(eachFilterRow + filterSize / 2) * filterSize + eachFilterColumn + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * imageColumns + index_x] = convolvedValue;
}

int main() {
    const int imageRows = 512;    // Adjust the number of image rows based on your data
    const int imageColumns = 512; // Adjust the number of image columns based on your data
    const int filterSize = 3;     // Adjust the filter size based on your requirement

    float *device_outputMatrix, *device_inputMatrix, *device_filter;

    // Allocate host memory
    float *host_outputMatrix = (float *)malloc(imageRows * imageColumns * sizeof(float));
    float *host_inputMatrix = (float *)malloc(imageRows * imageColumns * sizeof(float));
    float *host_filter = (float *)malloc(filterSize * filterSize * sizeof(float));

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < imageRows * imageColumns; ++i) {
        host_inputMatrix[i] = static_cast<float>(i);
    }

    for (int i = 0; i < filterSize * filterSize; ++i) {
        host_filter[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void **)&device_outputMatrix, imageRows * imageColumns * sizeof(float));
    cudaMalloc((void **)&device_inputMatrix, imageRows * imageColumns * sizeof(float));
    cudaMalloc((void **)&device_filter, filterSize * filterSize * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(device_inputMatrix, host_inputMatrix, imageRows * imageColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_filter, host_filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((imageColumns + block_size.x - 1) / block_size.x, (imageRows + block_size.y - 1) / block_size.y);

    // Launch the kernel
    convolution_kernel_v1<<<grid_size, block_size>>>(device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);

    // Copy data from device to host
    cudaMemcpy(host_outputMatrix, device_outputMatrix, imageRows * imageColumns * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(device_outputMatrix);
    cudaFree(device_inputMatrix);
    cudaFree(device_filter);
    free(host_outputMatrix);
    free(host_inputMatrix);
    free(host_filter);

    return 0;
}
 
