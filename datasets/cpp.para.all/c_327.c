#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Assuming you have a max function
float max(float a, float b) {
    return a > b ? a : b;
}

void softmax(float *x, int r, int c) {
    float temp1, temp2;
    for (int i = 0; i < r; i++) {
        temp1 = 0.;
        temp2 = 0.;
        for (int j = 0; j < c; j++) {
            temp1 = max(x[i * c + j], temp1);
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

int main() {
    // Define your array dimensions
    int r = 3;  // Replace with your actual row size
    int c = 4;  // Replace with your actual column size

    // Allocate memory for the array
    float *x = (float *)malloc(r * c * sizeof(float));

    // Initialize the array (example: filling with random values)
    for (int i = 0; i < r * c; i++) {
        x[i] = rand() % 100;  // Replace with your initialization logic
    }

    // Call the softmax function
    softmax(x, r, c);

    // Display the result (for demonstration purposes)
    printf("Softmax Result:\n");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%8.4f\t", x[i * c + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(x);

    return 0;
}
 
