#include <stdio.h>

// Function declaration
void mathKernel1(float *c, int size);

int main() {
    // Example data
    const int size = 5;
    float c[5];

    // Function call
    mathKernel1(c, size);

    // Output result
    printf("Resultant array:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", c[i]);
    }

    return 0;
}

// Function definition
void mathKernel1(float *c, int size) {
    int tid;
    float ia, ib;
    ia = ib = 0.0f;

    for (tid = 0; tid < size; tid++) {
        if (tid % 2 == 0) {
            ia = 100.0f;
        } else {
            ib = 200.0f;
        }
        c[tid] = ia + ib;
    }
}
 
