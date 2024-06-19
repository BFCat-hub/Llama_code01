#include <stdio.h>

// Function declaration
void transpose(int A[][10000], int trans[][10000]);

int main() {
    // Example data
    const int rows = 10000;
    const int cols = 10000;
    int A[rows][cols];
    int trans[cols][rows];

    // Initialize the array A with some values (for demonstration purposes)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i][j] = i * cols + j + 1; // Some arbitrary values
        }
    }

    // Function call
    transpose(A, trans);

    // Output result (printing a subset for demonstration purposes)
    printf("Original array A:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }

    printf("\nTransposed array:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", trans[i][j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void transpose(int A[][10000], int trans[][10000]) {
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 10000; j++) {
            trans[i][j] = A[j][i];
        }
    }
}
 
