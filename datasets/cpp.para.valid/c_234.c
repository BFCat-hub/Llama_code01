 #include <stdio.h>

// Function declaration
void doubleArraySign_cpu(double *d_in, double *d_out, int length);

int main() {
    // Example data
    const int length = 5;
    double d_in[] = {-2.5, 0.0, 1.5, -3.0, 2.0};
    double d_out[5];

    // Function call
    doubleArraySign_cpu(d_in, d_out, length);

    // Output result
    printf("Resultant array after doubleArraySign_cpu:\n");
    for (int i = 0; i < length; i++) {
        printf("%f ", d_out[i]);
    }

    return 0;
}

// Function definition
void doubleArraySign_cpu(double *d_in, double *d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = (0 < d_in[idx]) - (d_in[idx] < 0);
    }
}

