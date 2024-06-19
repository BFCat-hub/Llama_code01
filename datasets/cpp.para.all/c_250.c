#include <stdio.h>

// Function declaration
void cpuAdd(int *a, int *b, int *c, int vectorSize);

int main() {
    // Example data
    const int vectorSize = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    // Function call
    cpuAdd(a, b, c, vectorSize);

    // Output result
    printf("Resultant array:\n");
    for (int i = 0; i < vectorSize; i++) {
        printf("%d ", c[i]);
    }

    return 0;
}

// Function definition
void cpuAdd(int *a, int *b, int *c, int vectorSize) {
    #pragma omp parallel for
    for (int i = 0; i < vectorSize; i++) {
        c[i] = a[i] + b[i];
    }
}
 
