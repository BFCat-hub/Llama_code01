#include <stdio.h>
#include <stdlib.h>

// Function prototype
void SetToZero_kernel(float *d_vx, float *d_vy, float *d_vz, int w, int h, int l);

int main() {
    // Example data
    int w = 3; // Replace with the width of your data
    int h = 3; // Replace with the height of your data
    int l = 3; // Replace with the depth of your data

    // Allocate memory for data
    float *d_vx = (float *)malloc(w * h * l * sizeof(float));
    float *d_vy = (float *)malloc(w * h * l * sizeof(float));
    float *d_vz = (float *)malloc(w * h * l * sizeof(float));

    // Call the function
    SetToZero_kernel(d_vx, d_vy, d_vz, w, h, l);

    // Display the results (optional)
    // Note: Printing large data may not be practical

    // Free allocated memory
    free(d_vx);
    free(d_vy);
    free(d_vz);

    return 0;
}

// Function definition
void SetToZero_kernel(float *d_vx, float *d_vy, float *d_vz, int w, int h, int l) {
    unsigned int i, j;

    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            unsigned int index = j * w + i;
            for (int k = 0; k < l; ++k, index += w * h) {
                d_vx[index] = 0;
                d_vy[index] = 0;
                d_vz[index] = 0;
            }
        }
    }
}
 
