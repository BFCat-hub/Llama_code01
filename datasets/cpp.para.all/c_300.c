#include <stdio.h>

// Function prototype
void matrixMulOnHost(float *M, float *N, float *P, int width);

int main() {
    // Example data
    int width = 3;
    float M[] = {1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0,
                 7.0, 8.0, 9.0};
    float N[] = {9.0, 8.0, 7.0,
                 6.0, 5.0, 4.0,
                 3.0, 2.0, 1.0};
    float P[width * width];

    // Call the function
    matrixMulOnHost(M, N, P, width);

    // Display the results
    printf("Result Matrix:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", P[i * width + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void matrixMulOnHost(float *M, float *N, float *P, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            double sum = 0;
            for (int k = 0; k < width; ++k) {
                double a = M[i * width + k];
                double b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;
        }
    }
}
 
