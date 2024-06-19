#include <stdio.h>

// Function declaration
void clip_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

int main() {
    // Example data
    const int N = 5;
    const float ALPHA = 2.0;
    float X[] = {1.0, 3.0, 2.0, -1.0, 4.0};
    float Y[N];

    // Function call
    clip_cpu(N, ALPHA, X, 1, Y, 1);

    // Output result
    printf("Clipped array (Y): ");
    for (int i = 0; i < N; ++i) {
        printf("%f ", Y[i]);
    }

    return 0;
}

// Function definition
void clip_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    for (int i = 0; i < N; ++i) {
        float val = X[i * INCX];
        Y[i * INCY] = val > ALPHA ? val : 0;
    }
}
 
