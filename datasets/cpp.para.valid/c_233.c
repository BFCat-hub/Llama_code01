#include <stdio.h>

// Function declaration
void add_cpu(int N, int offset, float *X, int INCX);

int main() {
    // Example data
    const int N = 5;
    int offset = 10;
    float X[] = {-130.0, -129.0, -128.0, -127.0, -126.0};

    // Function call
    add_cpu(N, offset, X, 1);

    // Output result
    printf("Resultant array after add_cpu:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", X[i]);
    }

    return 0;
}

// Function definition
void add_cpu(int N, int offset, float *X, int INCX) {
    for (int i = 0; i < N; i++) {
        X[i * INCX] += offset;
        if (X[i * INCX] == -128.0f) {
            X[i * INCX] = -127.0f;
        }
    }
}
 
