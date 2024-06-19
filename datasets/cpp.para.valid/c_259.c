#include <stdio.h>

// Function declaration
void compareDoubleArrayToThreshold_cpu(double *d_in, int *d_out, int length, double threshold);

int main() {
    // Example data
    const int length = 5;
    double d_in[length] = {1.5, -2.0, 0.8, -0.3, 2.7};
    int d_out[length];

    // Function call
    double threshold = 1.0;
    compareDoubleArrayToThreshold_cpu(d_in, d_out, length, threshold);

    // Output result
    printf("Resultant d_out array:\n");
    for (int i = 0; i < length; ++i) {
        printf("%d ", d_out[i]);
    }

    return 0;
}

// Function definition
void compareDoubleArrayToThreshold_cpu(double *d_in, int *d_out, int length, double threshold) {
    for (int idx = 0; idx < length; idx++) {
        double abs_value = (d_in[idx] > 0) ? d_in[idx] : -d_in[idx];
        d_out[idx] = (abs_value < threshold) ? 1 : 0;
    }
}
 
