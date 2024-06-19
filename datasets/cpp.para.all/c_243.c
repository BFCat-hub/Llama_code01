#include <stdio.h>

// Function declaration
void doubleArrayVectorSubtract_cpu(double *d_in_a, double *d_in_b, double *d_out, int length);

int main() {
    // Example data
    const int length = 5;
    double d_in_a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double d_in_b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    double d_out[5];

    // Function call
    doubleArrayVectorSubtract_cpu(d_in_a, d_in_b, d_out, length);

    // Output result
    printf("Resultant d_out array:\n");
    for (int i = 0; i < length; i++) {
        printf("%f ", d_out[i]);
    }

    return 0;
}

// Function definition
void doubleArrayVectorSubtract_cpu(double *d_in_a, double *d_in_b, double *d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in_a[idx] - d_in_b[idx];
    }
}
 
