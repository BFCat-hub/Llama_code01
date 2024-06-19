#include <stdio.h>
#include <stdlib.h>

// Function prototype
void matrixMultiply_cpu(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns);

int main() {
    // Example data
    int numARows = 3;    // Replace with the number of rows in matrix A
    int numAColumns = 2; // Replace with the number of columns in matrix A
    int numBRows = 2;    // Replace with the number of rows in matrix B
    int numBColumns = 4; // Replace with the number of columns in matrix B

    // Allocate memory for matrices
    float *A = (float *)malloc(numARows * numAColumns * sizeof(float));
    float *B = (float *)malloc(numBRows * numBColumns * sizeof(float));
    float *C = (float *)malloc(numARows * numBColumns * sizeof(float));

    // Initialize matrices (for example)
    for (int i = 0; i < numARows * numAColumns; i++) {
        A[i] = i + 1.0; // Replace with your data
    }

    for (int i = 0; i < numBRows * numBColumns; i++) {
        B[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    matrixMultiply_cpu(A, B, C, numARows, numAColumns, numBRows, numBColumns);

    // Display the results (optional)
    // Note: Printing large data may not be practical

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}

// Function definition
void matrixMultiply_cpu(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns) {
    int numCRows = numARows;
    int numCColumns = numBColumns;

    for (int row = 0; row < numCRows; row++) {
        for (int col = 0; col < numCColumns; col++) {
            float sum = 0;

            for (int k = 0; k < numBRows; k++) {
                sum += A[row * numAColumns + k] * B[k * numBColumns + col];
            }

            C[row * numCColumns + col] = sum;
        }
    }
}
 
