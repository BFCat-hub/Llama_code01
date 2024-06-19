#include <stdio.h>
#include <stdlib.h>

// Function prototype
void boxesScale_cpu(const float *input, float *output, int dims, float scale0, float scale1, float scale2, float scale3);

int main() {
    // Example data
    int dims = 2;              // Replace with your data
    float scale0 = 2.0;        // Replace with your data
    float scale1 = 1.5;        // Replace with your data
    float scale2 = 3.0;        // Replace with your data
    float scale3 = 0.5;        // Replace with your data

    // Allocate memory for input and output
    float *input = (float *)malloc(dims * 4 * sizeof(float));
    float *output = (float *)malloc(dims * 4 * sizeof(float));

    // Initialize input (for example)
    for (int i = 0; i < dims * 4; i++) {
        input[i] = i + 1.0;     // Replace with your data
    }

    // Call the function
    boxesScale_cpu(input, output, dims, scale0, scale1, scale2, scale3);

    // Display the results (optional)
    printf("Output: ");
    for (int i = 0; i < dims * 4; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    // Free allocated memory
    free(input);
    free(output);

    return 0;
}

// Function definition
void boxesScale_cpu(const float *input, float *output, int dims, float scale0, float scale1, float scale2, float scale3) {
    for (int tid = 0; tid < dims; tid++) {
        output[tid * 4] = input[tid * 4] / scale0;
        output[tid * 4 + 1] = input[tid * 4 + 1] / scale1;
        output[tid * 4 + 2] = input[tid * 4 + 2] / scale2;
        output[tid * 4 + 3] = input[tid * 4 + 3] / scale3;
    }
}
 
