#include <stdio.h>

// Function prototype
void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c);

int main() {
    // Example data
    int n = 5;
    float a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float b[] = {6.0, 7.0, 8.0, 9.0, 10.0};
    float s[] = {0.2, 0.4, 0.6, 0.8, 1.0};
    float c[n];

    // Call the function
    weighted_sum_cpu(a, b, s, n, c);

    // Display the results
    printf("Weighted Sum:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", c[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c) {
    for (int i = 0; i < n; ++i) {
        c[i] = s[i] * a[i] + (1 - s[i]) * (b ? b[i] : 0);
    }
}
 
