#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Define the CUDA kernel
__global__ void cuInsertionSort(float *dist, long *ind, int width, int height, int k) {
    int l, i, j;
    float *p_dist;
    long *p_ind;
    float curr_dist, max_dist;
    long curr_row, max_row;
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (xIndex < width) {
        p_dist = dist + xIndex;
        p_ind = ind + xIndex;
        max_dist = p_dist[0];
        p_ind[0] = 1;

        for (l = 1; l < k; l++) {
            curr_row = l * width;
            curr_dist = p_dist[curr_row];

            if (curr_dist < max_dist) {
                i = l - 1;

                for (int a = 0; a < l - 1; a++) {
                    if (p_dist[a * width] > curr_dist) {
                        i = a;
                        break;
                    }
                }

                for (j = l; j > i; j--) {
                    p_dist[j * width] = p_dist[(j - 1) * width];
                    p_ind[j * width] = p_ind[(j - 1) * width];
                }

                p_dist[i * width] = curr_dist;
                p_ind[i * width] = l + 1;
            } else {
                p_ind[l * width] = l + 1;
            }

            max_dist = p_dist[curr_row];
        }

        max_row = (k - 1) * width;

        for (l = k; l < height; l++) {
            curr_dist = p_dist[l * width];

            if (curr_dist < max_dist) {
                i = k - 1;

                for (int a = 0; a < k - 1; a++) {
                    if (p_dist[a * width] > curr_dist) {
                        i = a;
                        break;
                    }
                }

                for (j = k - 1; j > i; j--) {
                    p_dist[j * width] = p_dist[(j - 1) * width];
                    p_ind[j * width] = p_ind[(j - 1) * width];
                }

                p_dist[i * width] = curr_dist;
                p_ind[i * width] = l + 1;
                max_dist = p_dist[max_row];
            }
        }
    }
}

// Host function to launch the CUDA kernel
void launchKernel(float *dist, long *ind, int width, int height, int k) {
    int blockSize = 256;
    int numBlocks = (width + blockSize - 1) / blockSize;

    cuInsertionSort<<<numBlocks, blockSize>>>(dist, ind, width, height, k);

    cudaDeviceSynchronize();

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(cuda_err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int k = 10;

    // Allocate memory on the host
    float *hostDist = (float *)malloc(width * height * sizeof(float));
    long *hostInd = (long *)malloc(width * height * sizeof(long));

    // Initialize hostDist and hostInd with your data

    // Allocate memory on the device
    float *deviceDist;
    long *deviceInd;

    cudaMalloc((void **)&deviceDist, width * height * sizeof(float));
    cudaMalloc((void **)&deviceInd, width * height * sizeof(long));

    // Transfer data from host to device
    cudaMemcpy(deviceDist, hostDist, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInd, hostInd, width * height * sizeof(long), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    launchKernel(deviceDist, deviceInd, width, height, k);

    // Transfer results from device to host
    cudaMemcpy(hostDist, deviceDist, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostInd, deviceInd, width * height * sizeof(long), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceDist);
    cudaFree(deviceInd);

    // Free host memory
    free(hostDist);
    free(hostInd);

    return 0;
}
 
