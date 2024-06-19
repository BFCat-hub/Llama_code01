#include <stdio.h>

// Function prototype
void MMDSelfComputeWithSum(float *x_average, int size_x, float *distance_matrix);

int main() {
    // Example data
    int size_x = 4;
    float x_average[] = {1.5, 2.0, 3.0, 4.5};
    float distance_matrix[size_x * size_x];

    // Call the function
    MMDSelfComputeWithSum(x_average, size_x, distance_matrix);

    // Display the results
    printf("Distance Matrix:\n");
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_x; j++) {
            printf("%.2f ", distance_matrix[i * size_x + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void MMDSelfComputeWithSum(float *x_average, int size_x, float *distance_matrix) {
    for (int i = 0; i < size_x; i++) {
        for (int j = i; j < size_x; j++) {
            distance_matrix[i * size_x + j] = x_average[i] * x_average[j];
            distance_matrix[j * size_x + i] = distance_matrix[i * size_x + j]; // Mirror elements
        }
    }
}
 
