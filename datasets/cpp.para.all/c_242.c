#include <stdio.h>

// Function declaration
void shiftIndices(long *vec_out, const long by, const long imageSize, const long N);

int main() {
    // Example data
    const long imageSize = 10;
    const long N = 5;
    const long by = 3;
    long vec_out[5];

    // Function call
    shiftIndices(vec_out, by, imageSize, N);

    // Output result
    printf("Resultant vec_out array:\n");
    for (int i = 0; i < N; i++) {
        printf("%ld ", vec_out[i]);
    }

    return 0;
}

// Function definition
void shiftIndices(long *vec_out, const long by, const long imageSize, const long N) {
    for (int idx = 0; idx < N; idx++) {
        vec_out[idx] = (imageSize + ((idx - N / 2 + by) % imageSize)) % imageSize;
    }
}
 
