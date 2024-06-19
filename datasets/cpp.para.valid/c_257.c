#include <stdio.h>

// Function declaration
void HammingDistanceCPU(int *c, const int *a, const int *b, long const int *size);

int main() {
    // Example data
    const int size = 5;
    int a[] = {1, 0, 1, 1, 0};
    int b[] = {0, 1, 1, 1, 1};
    int c = 0;

    // Function call
    HammingDistanceCPU(&c, a, b, &size);

    // Output result
    printf("Hamming distance: %d\n", c);

    return 0;
}

// Function definition
void HammingDistanceCPU(int *c, const int *a, const int *b, long const int *size) {
    for (int i = 0; i < *size; i += 1) {
        if (a[i] != b[i])
            *c = *c + 1;
    }
}
 
