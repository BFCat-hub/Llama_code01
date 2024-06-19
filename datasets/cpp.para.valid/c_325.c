#include <stdio.h>
#include <stdlib.h>

void CPU_array_rowKernel(double *input, double *output, int length) {
    int xCuda, yCuda;
    for (xCuda = 0; xCuda < length; xCuda++) {
        for (yCuda = 0; yCuda < length; yCuda++) {
            int idx = yCuda * length + xCuda;
            if (xCuda == 0 || xCuda == length - 1) {
                output[idx] = 0;
            } else {
                output[idx] = input[idx];
                output[idx] += xCuda == 0 ? 0 : input[idx - 1];
                output[idx] += xCuda == length - 1 ? 0 : input[idx + 1];
            }
        }
    }
}

int main() {
    // Define your array length
    int length = 5;

    // Allocate memory for input and output arrays
    double *input = (double *)malloc(length * length * sizeof(double));
    double *output = (double *)malloc(length * length * sizeof(double));

    // Initialize input array (example: filling with random values)
    for (int i = 0; i < length * length; i++) {
        input[i] = rand() % 100; // Replace with your initialization logic
    }

    // Call the CPU_array_rowKernel function
    CPU_array_rowKernel(input, output, length);

    // Display the result (for demonstration purposes)
    printf("Input Array:\n");
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            printf("%8.2f\t", input[length * i + j]);
        }
        printf("\n");
    }

    printf("\nOutput Array:\n");
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            printf("%8.2f\t", output[length * i + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(input);
    free(output);

    return 0;
}
 
