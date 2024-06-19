#include <stdio.h>

// Function declaration
void expandScoreFactors_cpu(const float *input, float *output, int dims, int clsNum);

int main() {
    // Example data
    const int dims = 8;
    const int clsNum = 2;
    float input[] = {1.0, 2.0, 3.0, 4.0};
    float output[8]; // Assuming dims elements

    // Function call
    expandScoreFactors_cpu(input, output, dims, clsNum);

    // Output result
    printf("Resultant expanded array:\n");
    for (int i = 0; i < dims; i++) {
        printf("%f ", output[i]);
    }

    return 0;
}

// Function definition
void expandScoreFactors_cpu(const float *input, float *output, int dims, int clsNum) {
    for (int tid = 0; tid < dims; tid++) {
        int k = tid / clsNum;
        output[tid] = input[k];
    }
}
 
