#include <stdio.h>

// Function declaration
void ReLU_forward(float *in, int *mask, int datasize, int training);

int main() {
    // Example data
    const int datasize = 5;
    float in[5] = {-1.5, 2.0, -0.8, 0.3, 1.7};
    int mask[5] = {0};  // Initialize mask array

    // Function call
    ReLU_forward(in, mask, datasize, 1);  // Training mode (mask updated)

    // Output result
    printf("Output array after ReLU forward pass:\n");
    for (int i = 0; i < datasize; ++i) {
        printf("%f ", in[i]);
    }

    printf("\nBinary mask (for training): ");
    for (int i = 0; i < datasize; ++i) {
        printf("%d ", mask[i]);
    }

    return 0;
}

// Function definition
void ReLU_forward(float *in, int *mask, int datasize, int training) {
    for (int i = 0; i < datasize; ++i) {
        int keep = in[i] > 0;
        if (training) {
            mask[i] = keep;
        }
        if (!keep) {
            in[i] = 0;
        }
    }
}
 
