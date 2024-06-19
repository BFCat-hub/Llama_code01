#include <stdio.h>
#include <stdlib.h>

// Function prototype
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);

int main() {
    // Example data
    int batch = 2;      // Replace with your data
    int filters = 3;    // Replace with your data
    int spatial = 4;    // Replace with your data

    // Allocate memory for x and mean
    float *x = (float *)malloc(batch * filters * spatial * sizeof(float));
    float *mean = (float *)malloc(filters * sizeof(float));

    // Initialize x (for example)
    for (int i = 0; i < batch * filters * spatial; i++) {
        x[i] = i + 1.0; // Replace with your data
    }

    // Call the function
    mean_cpu(x, batch, filters, spatial, mean);

    // Display the results (optional)
    printf("Mean: ");
    for (int i = 0; i < filters; i++) {
        printf("%f ", mean[i]);
    }
    printf("\n");

    // Free allocated memory
    free(x);
    free(mean);

    return 0;
}

// Function definition
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean) {
    float scale = 1.0 / (batch * spatial);
    int i, j, k;

    for (i = 0; i < filters; ++i) {
        mean[i] = 0;

        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }

        mean[i] *= scale;
    }
}
 
