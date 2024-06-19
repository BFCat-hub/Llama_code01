#include <stdio.h>
#include <math.h>

// Function declaration
void clamp_cpu(int N, float *X, int INCX, float clamp_min, float clamp_max);

int main() {
    // Example data
    const int N = 5;
    float X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const int INCX = 1;
    const float clamp_min = 2.0;
    const float clamp_max = 4.0;

    // Function call
    clamp_cpu(N, X, INCX, clamp_min, clamp_max);

    // Output result
    printf("Resultant array:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", X[i]);
    }

    return 0;
}

// Function definition
void clamp_cpu(int N, float *X, int INCX, float clamp_min, float clamp_max) {
    for (int i = 0; i < N; ++i) {
        X[i * INCX] = fmin(clamp_max, fmax(clamp_min, X[i * INCX]));
    }
}
 
