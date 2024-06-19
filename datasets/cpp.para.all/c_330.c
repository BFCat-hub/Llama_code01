#include <stdio.h>
#include <math.h>

void Softmax(float *x, const int r, const int c);

int main() {
    // Define array dimensions
    const int rows = 3;
    const int cols = 4;

    // Create a 2D array
    float array[rows][cols] = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0}
    };

    // Flatten the array for Softmax function
    float flatArray[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatArray[i * cols + j] = array[i][j];
        }
    }

    // Apply Softmax
    Softmax(flatArray, rows, cols);

    // Print the result
    printf("Softmax Result:\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", flatArray[i * cols + j]);
        }
        printf("\n");
    }

    return 0;
}

void Softmax(float *x, const int r, const int c) {
    float temp1, temp2;
    for (int i = 0; i < r; i++) {
        temp1 = 0.;
        temp2 = 0.;
        for (int j = 0; j < c; j++) {
            temp1 = fmaxf(x[i * c + j], temp1);
        }
        for (int j = 0; j < c; j++) {
            x[i * c + j] = expf(x[i * c + j] - temp1);
            temp2 += x[i * c + j];
        }
        for (int j = 0; j < c; j++) {
            x[i * c + j] /= temp2;
        }
    }
}
 
