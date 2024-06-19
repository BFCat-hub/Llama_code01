#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void subtractIntValues(int* destination, const int* value1, const int* value2, const unsigned int end) {
    unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (index = 0; index < end; index += stride) {
        destination[index] = value1[index] - value2[index];
    }
}

int main() {
    const unsigned int end = 512;

    int* h_value1 = (int*)malloc(end * sizeof(int));
    int* h_value2 = (int*)malloc(end * sizeof(int));
    int* h_destination = (int*)malloc(end * sizeof(int));

    for (unsigned int i = 0; i < end; ++i) {
        h_value1[i] = i;
        h_value2[i] = 2 * i;
    }

    int* d_value1;
    int* d_value2;
    int* d_destination;
    cudaMalloc((void**)&d_value1, end * sizeof(int));
    cudaMalloc((void**)&d_value2, end * sizeof(int));
    cudaMalloc((void**)&d_destination, end * sizeof(int));

    cudaMemcpy(d_value1, h_value1, end * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value2, h_value2, end * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((end + blockSize.x - 1) / blockSize.x, 1);

    subtractIntValues<<<gridSize, blockSize>>>(d_destination, d_value1, d_value2, end);

    cudaMemcpy(h_destination, d_destination, end * sizeof(int), cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < 10; ++i) {
        printf("h_destination[%u]: %d\n", i, h_destination[i]);
    }

    free(h_value1);
    free(h_value2);
    free(h_destination);
    cudaFree(d_value1);
    cudaFree(d_value2);
    cudaFree(d_destination);

    return 0;
}