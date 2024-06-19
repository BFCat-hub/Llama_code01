#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Define the CUDA kernel
__global__ void cuda_Cross_py_forward_A_kernel(float *logits_data, float *logits_grad, bool training, int num_classes, int *truth, int *count, float *thread_loss, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        if (truth[i] < 0) {
            count[i] = 0;
            return;
        }

        float *logit = &logits_data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;

        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);

        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }

        if (training) {
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits_grad[i * num_classes + j] = prob;
            }
            logits_grad[i * num_classes + truth[i]] -= 1.0;
        }

        count[i] = 1;
        thread_loss[i] = logf(sum_exp) - logit[truth[i]];
    }
}

// Host function to call the CUDA kernel
void launch_kernel(float *logits_data, float *logits_grad, bool training, int num_classes, int *truth, int *count, float *thread_loss, int size) {
    // Calculate grid and block sizes based on your problem
    dim3 gridSize(<<<YourGridSizeX>>>, <<<YourGridSizeY>>>);
    dim3 blockSize(<<<YourBlockSizeX>>>, <<<YourBlockSizeY>>>;

    // Launch the kernel
    cuda_Cross_py_forward_A_kernel<<<gridSize, blockSize>>>(logits_data, logits_grad, training, num_classes, truth, count, thread_loss, size);

    // Synchronize to wait for the kernel to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    }
}

int main() {
    // Example usage
    int size = 100;  // Set your array size
    float *logits_data, *logits_grad, *thread_loss;
    int *truth, *count;

    // Allocate memory on the GPU
    cudaMalloc((void **)&logits_data, size * sizeof(float));
    cudaMalloc((void **)&logits_grad, size * sizeof(float));
    cudaMalloc((void **)&truth, size * sizeof(int));
    cudaMalloc((void **)&count, size * sizeof(int));
    cudaMalloc((void **)&thread_loss, size * sizeof(float));

    // Initialize your input arrays (logits_data, truth, etc.)

    // Example: call the CUDA kernel
    launch_kernel(logits_data, logits_grad, true, 10, truth, count, thread_loss, size);

    // Free allocated memory on the GPU
    cudaFree(logits_data);
    cudaFree(logits_grad);
    cudaFree(truth);
    cudaFree(count);
    cudaFree(thread_loss);

    return 0;
}
 
