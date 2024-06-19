#include <stdio.h>

// Function prototype
void flipKernel(float *array1, int width);

int main() {
    // Example data
    int width = 4;
    float array1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};

    // Call the function
    flipKernel(array1, width);

    // Display the results
    printf("Flipped Array:\n");
    for (int i = 0; i < width * width; i++) {
        printf("%.1f ", array1[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void flipKernel(float *array1, int width) {
    for (int current_index = 0; current_index < width * width / 2; current_index++) {
        int replace = (width - 1 - current_index / width) * width + current_index % width;
        float temp = array1[current_index];
        array1[current_index] = array1[replace];
        array1[replace] = temp;
    }
}
 
