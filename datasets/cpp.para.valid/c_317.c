#include <stdio.h>
#include <stdlib.h>

// Function prototype
void backward_avgpool_layer(int batch, int c, int w, int h, float *delta);

int main() {
    // Example data
    int batch = 2; // Replace with your data
    int c = 3;    // Replace with your data
    int w = 4;    // Replace with your data
    int h = 4;    // Replace with your data

    // Allocate memory for delta
    float *delta = (float *)malloc(batch * c * w * h * sizeof(float));

    // Initialize delta (for example)
    for (int i = 0; i < batch * c * w * h; i++) {
        delta[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    backward_avgpool_layer(batch, c, w, h, delta);

    // Display the results (optional)
    // Note: Printing large data may not be practical

    // Free allocated memory
    free(delta);

    return 0;
}

// Function definition
void backward_avgpool_layer(int batch, int c, int w, int h, float *delta) {
    int b, i, k;

    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            int out_index = k + b * c;

            for (i = 0; i < h * w; ++i) {
                int in_index = i + h * w * (k + b * c);
                delta[in_index] += delta[out_index] / (h * w);
            }
        }
    }
}
 
