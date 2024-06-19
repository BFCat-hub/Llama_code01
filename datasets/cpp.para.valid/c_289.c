#include <stdio.h>

// Function prototype
void roundOff(float *mat, int N, int M);

int main() {
    // Example data
    int N = 3;
    int M = 4;
    float mat[] = {1.2, -2.5, 3.8,
                   -4.1, 5.6, -6.9,
                   7.4, -8.2, 9.0,
                   -10.3, 11.7, -12.5};

    // Call the function
    roundOff(mat, N, M);

    // Display the results
    printf("Rounded Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f ", mat[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void roundOff(float *mat, int N, int M) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (mat[i * N + j] >= 0)
                mat[i * N + j] = (int)(mat[i * N + j] + 0.5);
            else
                mat[i * N + j] = (int)(mat[i * N + j] - 0.5);
        }
    }
}
 
