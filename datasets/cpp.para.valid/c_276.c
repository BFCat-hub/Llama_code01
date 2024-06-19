#include <stdio.h>
#include <math.h>

// Function prototype
void kernel(float *x, int n);

int main() {
    // Example data
    int n = 5;
    float x[n];

    // Call the function
    kernel(x, n);

    // Display the results
    printf("Resultant x:\n");
    for (int i = 0; i < n; i++) {
        printf("%.4f ", x[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void kernel(float *x, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < 1000; j++) {
            sum += sqrt(pow(3.14159, i)) / (float)j;
        }
        x[i] = sum;
    }
}
 
