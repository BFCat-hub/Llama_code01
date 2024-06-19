#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void flatten(float *x, int size, int layers, int batch, int forward) {
    float *swap = calloc(size * layers * batch, sizeof(float));
    int i, c, b;
    for (b = 0; b < batch; ++b) {
        for (c = 0; c < layers; ++c) {
            for (i = 0; i < size; ++i) {
                int i1 = b * layers * size + c * size + i;
                int i2 = b * layers * size + i * layers + c;
                if (forward)
                    swap[i2] = x[i1];
                else
                    swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size * layers * batch * sizeof(float));
    free(swap);
}

int main() {
    // Define array dimensions
    int size = 2;
    int layers = 3;
    int batch = 4;

    // Create a 3D array
    float *array = (float *)malloc(size * layers * batch * sizeof(float));

    // Initialize the array (for demonstration purposes)
    for (int i = 0; i < size * layers * batch; ++i) {
        array[i] = i + 1;
    }

    // Print the original array
    printf("Original Array:\n");
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < layers; ++c) {
            for (int i = 0; i < size; ++i) {
                printf("%8.2f ", array[b * layers * size + c * size + i]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Flatten the array (forward direction)
    flatten(array, size, layers, batch, 1);

    // Print the flattened array
    printf("\nFlattened Array (Forward):\n");
    for (int i = 0; i < size * layers * batch; ++i) {
        printf("%8.2f ", array[i]);
    }
    printf("\n");

    // Unflatten the array (backward direction)
    flatten(array, size, layers, batch, 0);

    // Print the unflattened array
    printf("\nUnflattened Array (Backward):\n");
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < layers; ++c) {
            for (int i = 0; i < size; ++i) {
                printf("%8.2f ", array[b * layers * size + c * size + i]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Free allocated memory
    free(array);

    return 0;
}
 
