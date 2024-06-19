#include <stdio.h>

void conv2_cpu(float *A, float *kernel, int inputSize, int depth, int kernelSize, int stride, int pad, float *B, int outputSize) {
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            int Ai = i * stride;
            int Aj = j * stride;
            int startk = (pad - Ai) < 0 ? 0 : pad - Ai;
            int endk = kernelSize < (inputSize + pad - Ai) ? kernelSize : (inputSize + pad - Ai);
            int startl = (pad - Aj) < 0 ? 0 : pad - Aj;
            int endl = kernelSize < (inputSize + pad - Aj) ? kernelSize : (inputSize + pad - Aj);
            float sum = 0;
            for (int d = 0; d < depth; d++) {
                for (int k = startk; k < endk; k++) {
                    for (int l = startl; l < endl; l++) {
                        sum += A[d * inputSize * inputSize + (Ai + k - pad) * inputSize + Aj + l - pad] * kernel[d * kernelSize * kernelSize + k * kernelSize + l];
                    }
                }
                B[d * outputSize * outputSize + i * outputSize + j] = sum;
            }
            B[i * outputSize + j] = sum;
        }
    }
}

int main() {
    // Test conv2_cpu function with a simple example
    int inputSize = 4;
    int depth = 3;
    int kernelSize = 3;
    int stride = 1;
    int pad = 1;
    int outputSize = (inputSize - kernelSize + 2 * pad) / stride + 1;

    float A[3 * 4 * 4]; // Assuming depth = 3, inputSize = 4
    float kernel[3 * 3 * 3]; // Assuming depth = 3, kernelSize = 3
    float B[3 * 3 * 3]; // Assuming depth = 3, outputSize = 3

    // Initialize A and kernel with appropriate values

    // Call the conv2_cpu function
    conv2_cpu(A, kernel, inputSize, depth, kernelSize, stride, pad, B, outputSize);

    // Display the result matrix B
    for (int d = 0; d < depth; d++) {
        printf("Depth %d:\n", d);
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                printf("%f ", B[d * outputSize * outputSize + i * outputSize + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
 
