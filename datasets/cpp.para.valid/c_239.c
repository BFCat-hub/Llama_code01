#include <stdio.h>

// Function declaration
void sumRowKernel_cpu(const int *d_in, int *d_out, int DIM);

int main() {
    // Example data
    const int DIM = 4;
    int d_in[] = {1, 2, 3, 4};
    int d_out;

    // Function call
    sumRowKernel_cpu(d_in, &d_out, DIM);

    // Output result
    printf("Sum of elements in the row: %d\n", d_out);

    return 0;
}

// Function definition
void sumRowKernel_cpu(const int *d_in, int *d_out, int DIM) {
    int sum = 0;

    for (int i = 0; i < DIM; i++) {
        sum += d_in[i];
    }

    *d_out = sum;
}
 
