#include <stdio.h>

// Function prototype
void Transpose2d(float *array_transpose, float *array, const int r, const int c);

int main() {
    // Example data
    int r = 3;
    int c = 4;
    float array[] = {1.0, 2.0, 3.0, 4.0,
                     5.0, 6.0, 7.0, 8.0,
                     9.0, 10.0, 11.0, 12.0};
    float array_transpose[c * r];

    // Call the function
    Transpose2d(array_transpose, array, r, c);

    // Display the results
    printf("Original Array:\n");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.2f ", array[i * c + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Array:\n");
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < r; j++) {
            printf("%.2f ", array_transpose[i * r + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void Transpose2d(float *array_transpose, float *array, const int r, const int c) {
    int i, j;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            array_transpose[j * r + i] = array[i * c + j];
        }
    }
}
 
