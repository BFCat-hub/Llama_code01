#include <stdio.h>
#include <stdlib.h>

// Function prototype
void expandBoxes_cpu(const float *input, float *output, int dims, int clsNum);

int main() {
    // Example data
    int dims = 12;
    int clsNum = 3;
    float *input = (float *)malloc(clsNum * 4 * sizeof(float));
    float *output = (float *)malloc(dims * 4 * sizeof(float));

    // Initialize input matrix (for example)
    for (int i = 0; i < clsNum * 4; i++) {
        input[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    expandBoxes_cpu(input, output, dims, clsNum);

    // Display the results
    printf("Expanded Boxes Matrix:\n");
    for (int i = 0; i < dims; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", output[i * 4 + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(input);
    free(output);

    return 0;
}

// Function definition
void expandBoxes_cpu(const float *input, float *output, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        int k = tid / clsNum;
        output[tid * 4 + 0] = input[k * 4 + 0];
        output[tid * 4 + 1] = input[k * 4 + 1];
        output[tid * 4 + 2] = input[k * 4 + 2];
        output[tid * 4 + 3] = input[k * 4 + 3];
    }
}
 
