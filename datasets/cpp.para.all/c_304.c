#include <stdio.h>

// Function prototype
void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);

int main() {
    // Example data
    int NX = 3;
    int NY = 2;
    int B = 4;
    float X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float Y[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float OUT[NX + NY];

    // Call the function
    inter_cpu(NX, X, NY, Y, B, OUT);

    // Display the results
    printf("Interleaved Array:\n");
    for (int i = 0; i < NX + NY; i++) {
        printf("%.2f ", OUT[i]);
    }

    return 0;
}

// Function definition
void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT) {
    int i, j;
    int index = 0;
    for (j = 0; j < B; ++j) {
        for (i = 0; i < NX; ++i) {
            OUT[index++] = X[j * NX + i];
        }
        for (i = 0; i < NY; ++i) {
            OUT[index++] = Y[j * NY + i];
        }
    }
}
 
