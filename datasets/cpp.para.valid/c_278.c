#include <stdio.h>

// Function prototype
void matVecRowSub_cpu(const double *mat, const double *vec, double *buf, int m, int n);

int main() {
    // Example data
    int m = 3;
    int n = 4;
    double mat[] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double vec[] = {2.0, 4.0, 6.0, 8.0};
    double buf[m * n];

    // Call the function
    matVecRowSub_cpu(mat, vec, buf, m, n);

    // Display the results
    printf("Resultant Buffer:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", buf[i * n + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void matVecRowSub_cpu(const double *mat, const double *vec, double *buf, int m, int n) {
    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        buf[i * n + j] = mat[i * n + j] - vec[j];
    }
}
 
