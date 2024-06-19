#include <stdio.h>

// Function declaration
void addVectorsInto_cpu(float *result, const float *a, const float *b, int N);

int main() {
    // Example data
    const int N = 5;
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float result[5];

    // Function call
    addVectorsInto_cpu(result, a, b, N);

    // Output result
    printf("Resultant vector after addition:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", result[i]);
    }

    return 0;
}

// Function definition
void addVectorsInto_cpu(float *result, const float *a, const float *b, int N) {
    for (int i = 0; i < N; i++) {
        result[i] = a[i] + b[i];
    }
}
 
