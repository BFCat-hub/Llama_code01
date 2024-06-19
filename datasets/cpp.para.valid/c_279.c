#include <stdio.h>

// Function prototype
void transKernel(float *array1, float *array2, int width);

int main() {
    // Example data
    int width = 3;
    float array1[] = {1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0,
                      7.0, 8.0, 9.0};
    float array2[width * width];

    // Call the function
    transKernel(array1, array2, width);

    // Display the results
    printf("Transposed Array:\n");
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            printf("%.2f ", array2[x * width + y]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void transKernel(float *array1, float *array2, int width) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            int current_index = x * width + y;
            int replace = y * width + x;
            array2[replace] = array1[current_index];
        }
    }
}
 
