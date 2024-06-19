#include <stdio.h>
#include <stdlib.h>

// Function prototype
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);

int main() {
    // Example data
    int NX = 3; // Replace with the size of X dimension
    int NY = 2; // Replace with the size of Y dimension
    int B = 4;  // Replace with the batch size

    // Allocate memory for data
    float *X = (float *)malloc(B * NX * sizeof(float));
    float *Y = (float *)malloc(B * NY * sizeof(float));
    float *OUT = (float *)malloc((NX + NY) * B * sizeof(float));

    // Initialize input matrices (for example)
    for (int i = 0; i < B * NX; i++) {
        X[i] = i + 1.0; // Replace with your data
    }

    for (int i = 0; i < B * NY; i++) {
        Y[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    deinter_cpu(NX, X, NY, Y, B, OUT);

    // Display the results (optional)
    // Note: Printing large data may not be practical

    // Free allocated memory
    free(X);
    free(Y);
    free(OUT);

    return 0;
}

// Function definition
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT) {
    int i, j;
    int index = 0;

    for (j = 0; j < B; ++j) {
        for (i = 0; i < NX; ++i) {
            if (X) X[j * NX + i] += OUT[index];
            ++index;
        }

        for (i = 0; i < NY; ++i) {
            if (Y) Y[j * NY + i] += OUT[index];
            ++index;
        }
    }
}
 
