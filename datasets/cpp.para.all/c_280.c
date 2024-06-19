#include <stdio.h>

// Function prototype
void rowSumSquare_cpu(const double *mat, double *buf, int m, int n);

int main() {
    // Example data
    int m = 3;
    int n = 4;
    double mat[] = {1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0};
    double buf[m];

    // Call the function
    rowSumSquare_cpu(mat, buf, m, n);

    // Display the results
    printf("Row Sum Squares:\n");
    for (int i = 0; i < m; i++) {
        printf("%.2f ", buf[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void rowSumSquare_cpu(const double *mat, double *buf, int m, int n) {
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
   
 
