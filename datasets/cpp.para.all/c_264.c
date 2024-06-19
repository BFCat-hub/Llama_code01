#include <stdio.h>

// Function declaration
void update_x(double *x, double *a, double *b, int n);

int main() {
    // Example data
    const int n = 5;
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double a[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    double b[] = {1.0, 2.0, 3.0, 2.0, 1.0};

    // Function call
    update_x(x, a, b, n);

    // Output result
    printf("Updated array (x): ");
    for (int i = 0; i < n; ++i) {
        printf("%f ", x[i]);
    }

    return 0;
}

// Function definition
void update_x(double *x, double *a, double *b, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = (2.0 / 3.0) * (a[i] / b[i]) + (1.0 / 3.0) * x[i];
    }
}
 
