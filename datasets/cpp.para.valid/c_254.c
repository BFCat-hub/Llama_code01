 #include <stdio.h>

// Function declaration
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);

int main() {
    // Example data
    const int N = 5;
    const float ALPHA = 2.0;
    float X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const int INCX = 1;
    float Y[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    const int INCY = 1;

    // Function call
    axpy_cpu(N, ALPHA, X, INCX, Y, INCY);

    // Output result
    printf("Resultant array Y after axpy operation:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", Y[i]);
    }

    return 0;
}

// Function definition
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] += ALPHA * X[i * INCX];
    }
}

