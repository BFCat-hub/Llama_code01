#include <stdio.h>
#include <math.h>

// Function prototype
float CEE(float *x, int *t, int r, int c);

int main() {
    // Example data
    int r = 2;
    int c = 3;
    float x[] = {0.2, 0.8, 0.6, 0.4, 0.5, 0.9};
    int t[] = {0, 1, 1, 0, 1, 0};

    // Call the function
    float result = CEE(x, t, r, c);

    // Display the result
    printf("Cross-Entropy Error: %.4f\n", result);

    return 0;
}

// Function definition
float CEE(float *x, int *t, int r, int c) {
    float temp = 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            if (t[i * c + j] == 1) {
                temp += log(x[i * c + j] + 1e-7);
                continue;
            }
        }
    }
    temp /= -r;
    return temp;
}
 
