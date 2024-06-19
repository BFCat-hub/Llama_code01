#include <stdio.h>

// Function declaration
void sum_backward(float *db, float *dout, int r, int c);

int main() {
    // Example data
    const int r = 3;
    const int c = 2;
    float db[2] = {0.0, 0.0};
    float dout[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Function call
    sum_backward(db, dout, r, c);

    // Output result
    printf("Gradients with respect to bias terms (db): ");
    for (int i = 0; i < c; ++i) {
        printf("%f ", db[i]);
    }

    return 0;
}

// Function definition
void sum_backward(float *db, float *dout, int r, int c) {
    for (int j = 0; j < c; ++j) {
        for (int i = 0; i < r; ++i) {
            db[j] += dout[i * c + j];
        }
    }
}
 
