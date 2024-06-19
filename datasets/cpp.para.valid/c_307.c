#include <stdio.h>
#include <stdlib.h>

// Function prototype
void matrixProduct(double *matrix_a, double *matrix_b, double *matrix_c, int width, int height, int from, int my_rank);

int main() {
    // Example data
    int width = 3;
    int height = 3;
    int from = 0;
    int my_rank = 0;
    
    double *matrix_a = (double *)malloc(width * height * sizeof(double));
    double *matrix_b = (double *)malloc(width * height * sizeof(double));
    double *matrix_c = (double *)malloc(width * height * sizeof(double));

    // Initialize matrices (for example)
    for (int i = 0; i < width * height; i++) {
        matrix_a[i] = i + 1.0; // Replace with your data
        matrix_b[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    matrixProduct(matrix_a, matrix_b, matrix_c, width, height, from, my_rank);

    // Display the results
    printf("Result Matrix:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%.2f ", matrix_c[i * width + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(matrix_a);
    free(matrix_b);
    free(matrix_c);

    return 0;
}

// Function definition
void matrixProduct(double *matrix_a, double *matrix_b, double *matrix_c, int width, int height, int from, int my_rank) {
    int row, col;

    for (row = 0; row < width; row++) {
        for (col = 0; col < height; col++) {
            matrix_c[row * width + col] = 0;

            for (int k = 0; k < width; k++) {
                matrix_c[row * width + col] += matrix_a[((row + from) * width) + k] * matrix_b[k * width + col];
            }
        }
    }
}
 
