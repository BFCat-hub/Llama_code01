#include <stdio.h>
#include <stdlib.h>

// Function prototype
void multiply_matrices(float *A_Matrix, float *B_Matrix, float *ANS_Matrix, int N);

int main() {
    // Example data
    int N = 3; // Replace with the size of your matrices
    float *A_Matrix = (float *)malloc(N * N * sizeof(float));
    float *B_Matrix = (float *)malloc(N * N * sizeof(float));
    float *ANS_Matrix = (float *)malloc(N * N * sizeof(float));

    // Initialize input matrices (for example)
    for (int i = 0; i < N * N; i++) {
        A_Matrix[i] = i + 1.0; // Replace with your data
        B_Matrix[i] = i + 2.0; // Replace with your data
    }

    // Call the function
    multiply_matrices(A_Matrix, B_Matrix, ANS_Matrix, N);

    // Display the results
    printf("Resultant Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", ANS_Matrix[i * N + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(A_Matrix);
    free(B_Matrix);
    free(ANS_Matrix);

    return 0;
}

// Function definition
void multiply_matrices(float *A_Matrix, float *B_Matrix, float *ANS_Matrix, int N) {
    int i, j, k;
    float sum, m, n;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0;
            for (k = 0; k < N; k++) {
                m = *(A_Matrix + i * N + k);
                n = *(B_Matrix + k * N + j);
                sum += m * n;
            }
            *(ANS_Matrix + i * N + j) = sum;
        }
    }
}
 
