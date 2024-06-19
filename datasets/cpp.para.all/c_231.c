#include <stdio.h>

// Function declaration
void add_vec_scalaire_cpu(int *vec, int *res, int a, long N);

int main() {
    // Example data
    const long N = 5;
    int vec[] = {1, 2, 3, 4, 5};
    int res[5];
    int scalar = 10;

    // Function call
    add_vec_scalaire_cpu(vec, res, scalar, N);

    // Output result
    printf("Resultant array after adding scalar to vector:\n");
    for (long i = 0; i < N; i++) {
        printf("%d ", res[i]);
    }

    return 0;
}

// Function definition
void add_vec_scalaire_cpu(int *vec, int *res, int a, long N) {
    for (long i = 0; i < N; i++) {
        res[i] = vec[i] + a;
    }
}
 
