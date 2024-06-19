#include <stdio.h>

// Function prototype
void vectorMatrixMult(long int totalPixels, float *matrix, float *vector, float *out);

int main() {
    // Example data
    long int totalPixels = 3;
    float matrix[] = {1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 9.0};
    float vector[] = {2.0, 1.0, 3.0};
    float out[totalPixels];

    // Call the function
    vectorMatrixMult(totalPixels, matrix, vector, out);

    // Display the results
    printf("Resultant Vector:\n");
    for (long int i = 0; i < totalPixels; i++) {
        printf("%.2f ", out[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void vectorMatrixMult(long int totalPixels, float *matrix, float *vector, float *out) {
    for (long int i = 0; i < totalPixels; i++) {
        float sum = 0.0;
        for (long int j = 0; j < totalPixels; j++) {
            sum += matrix[i * totalPixels + j] * vector[j];
        }
        out[i] = sum;
    }
}
 
