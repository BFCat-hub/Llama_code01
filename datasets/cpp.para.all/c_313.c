#include <stdio.h>
#include <stdlib.h>

// Function prototype
void solveLower_cpu(const double *lower, const double *b, double *buf, int dim, int n);

int main() {
    // Example data
    int dim = 3; // Replace with the dimension of your data
    int n = 2;   // Replace with the number of data sets

    // Allocate memory for data
    double *lower = (double *)malloc(dim * dim * sizeof(double));
    double *b = (double *)malloc(n * dim * sizeof(double));
    double *buf = (double *)malloc(n * dim * sizeof(double));

    // Initialize input matrices (for example)
    for (int i = 0; i < dim * dim; i++) {
        lower[i] = i + 1.0; // Replace with your data
    }

    for (int i = 0; i < n * dim; i++) {
        b[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    solveLower_cpu(lower, b, buf, dim, n);

    // Display the results (optional)
    // Note: Printing large data may not be practical

    // Free allocated memory
    free(lower);
    free(b);
    free(buf);

    return 0;
}

// Function definition
void solveLower_cpu(const double *lower, const double *b, double *buf, int dim, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < dim; i++) {
            double val = b[k * dim + i];
            for (int j = 0; j < i; j++) {
                val -= lower[i * dim + j] * buf[k * dim + j];
            }
            buf[k * dim + i] = val / lower[i * dim + i];
        }
    }
}
 
