#include <stdio.h>

// Function prototype
void mean(float *A, float *means, int size_row, int size_col);

int main() {
    // Example data
    int size_row = 3;
    int size_col = 4;
    float A[] = {1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0};
    float means[size_col];

    // Call the function
    mean(A, means, size_row, size_col);

    // Display the means
    printf("Means:\n");
    for (int i = 0; i < size_col; i++) {
        printf("%.2f ", means[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void mean(float *A, float *means, int size_row, int size_col) {
    for (int idx = 0; idx < size_col; idx++) {
        for (int i = 0; i < size_row; i++) {
            means[idx] += A[idx * size_row + i];
        }
        means[idx] = means[idx] / size_row;
    }
}
 
