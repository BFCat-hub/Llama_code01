#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256

__global__ void get_positive_data(const float *all_box, const float *all_scores, const float *all_conf,
                                   const int *conf_inds, float *positive_box, float *positive_scores,
                                   float *positive_conf, int dims, int clsNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }

    if (conf_inds[tid] != (-1)) {
        positive_box[tid * 4 + 0] = all_box[tid * 4 + 0];
        positive_box[tid * 4 + 1] = all_box[tid * 4 + 1];
        positive_box[tid * 4 + 2] = all_box[tid * 4 + 2];
        positive_box[tid * 4 + 3] = all_box[tid * 4 + 3];

        for (int i = 0; i < clsNum; i++) {
            positive_scores[tid * clsNum + i] = all_scores[tid * clsNum + i];
        }

        positive_conf[tid] = all_conf[tid];
    } else {
        positive_box[tid * 4 + 0] = 0;
        positive_box[tid * 4 + 1] = 0;
        positive_box[tid * 4 + 2] = 0;
        positive_box[tid * 4 + 3] = 0;

        for (int i = 0; i < clsNum; i++) {
            positive_scores[tid * clsNum + i] = (-1);
        }

        positive_conf[tid] = (-1);
    }
}

int main() {
    // Example usage
    int dims = 1000;
    int clsNum = 5;

    // Allocate memory on the host
    float *all_box_host = (float *)malloc(dims * 4 * sizeof(float));
    float *all_scores_host = (float *)malloc(dims * clsNum * sizeof(float));
    float *all_conf_host = (float *)malloc(dims * sizeof(float));
    int *conf_inds_host = (int *)malloc(dims * sizeof(int));

    float *positive_box_host = (float *)malloc(dims * 4 * sizeof(float));
    float *positive_scores_host = (float *)malloc(dims * clsNum * sizeof(float));
    float *positive_conf_host = (float *)malloc(dims * sizeof(float));

    // Initialize input data (all_box, all_scores, all_conf, conf_inds) on the host

    // Allocate memory on the device
    float *all_box_device, *all_scores_device, *all_conf_device;
    int *conf_inds_device;
    float *positive_box_device, *positive_scores_device, *positive_conf_device;

    cudaMalloc((void **)&all_box_device, dims * 4 * sizeof(float));
    cudaMalloc((void **)&all_scores_device, dims * clsNum * sizeof(float));
    cudaMalloc((void **)&all_conf_device, dims * sizeof(float));
    cudaMalloc((void **)&conf_inds_device, dims * sizeof(int));

    cudaMalloc((void **)&positive_box_device, dims * 4 * sizeof(float));
    cudaMalloc((void **)&positive_scores_device, dims * clsNum * sizeof(float));
    cudaMalloc((void **)&positive_conf_device, dims * sizeof(float));

    // Copy input data from host to device

    // Launch the CUDA kernel
    dim3 gridDim((dims + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);

    get_positive_data<<<gridDim, blockDim>>>(all_box_device, all_scores_device, all_conf_device,
                                             conf_inds_device, positive_box_device,
                                             positive_scores_device, positive_conf_device, dims,
                                             clsNum);

    // Copy the result back from device to host

    // Free allocated memory on both host and device

    free(all_box_host);
    free(all_scores_host);
    free(all_conf_host);
    free(conf_inds_host);

    free(positive_box_host);
    free(positive_scores_host);
    free(positive_conf_host);

    cudaFree(all_box_device);
    cudaFree(all_scores_device);
    cudaFree(all_conf_device);
    cudaFree(conf_inds_device);

    cudaFree(positive_box_device);
    cudaFree(positive_scores_device);
    cudaFree(positive_conf_device);

    return 0;
}
 
