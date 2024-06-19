#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function prototype
void binarize_input(float *input, int n, int size, float *binary);

int main() {
    // Example data
    int n = 3;    // Replace with the size of input dimension
    int size = 5; // Replace with the size of size dimension

    // Allocate memory for data
    float *input = (float *)malloc(n * size * sizeof(float));
    float *binary = (float *)malloc(n * size * sizeof(float));

    // Initialize input matrix (for example)
    for (int i = 0; i < n * size; i++) {
        input[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    binarize_input(input, n, size, binary);

    // Display the results (optional)
    // Note: Printing large data may not be practical

    // Free allocated memory
    free(input);
    free(binary);

    return 0;
}

// Function definition
void binarize_input(float *input, int n, int size, float *binary) {
    int i, s;

    for (s = 0; s < size; ++s) {
        float mean = 0;

        for (i = 0; i < n; ++i) {
            mean += fabs(input[i * size + s]);
        }

        mean = mean / n;

        for (i = 0; i < n; ++i) {
            binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
        }
    }
}
 
