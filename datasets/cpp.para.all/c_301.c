#include <stdio.h>
#include <math.h>

// Function prototype
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);

int main() {
    // Example data
    int batch = 2;
    int filters = 3;
    int spatial = 4;
    float x[] = {1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0,
                 13.0, 14.0, 15.0, 16.0,
                 17.0, 18.0, 19.0, 20.0,
                 21.0, 22.0, 23.0, 24.0};
    float mean[] = {2.0, 4.0, 6.0};
    float variance[] = {1.0, 2.0, 3.0};

    // Call the function
    normalize_cpu(x, mean, variance, batch, filters, spatial);

    // Display the results
    printf("Normalized Data:\n");
    for (int b = 0; b < batch; ++b) {
        for (int f = 0; f < filters; ++f) {
            for (int i = 0; i < spatial; ++i) {
                int index = b * filters * spatial + f * spatial + i;
                printf("%.4f ", x[index]);
            }
            printf("\n");
        }
    }

    return 0;
}

// Function definition
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial) {
    for (int b = 0; b < batch; ++b) {
        for (int f = 0; f < filters; ++f) {
            for (int i = 0; i < spatial; ++i) {
                int index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + 0.000001f);
            }
        }
    }
}
 
