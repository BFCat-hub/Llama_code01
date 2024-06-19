#include <stdio.h>

// Function declaration
void binarize_cpu(float *input, int n, float *binary);

int main() {
    // Example data
    const int n = 5;
    float input[] = {1.0, -2.0, 3.0, -4.0, 5.0};
    float binary[5];

    // Function call
    binarize_cpu(input, n, binary);

    // Output result
    printf("Resultant array after binarization:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", binary[i]);
    }

    return 0;
}

// Function definition
void binarize_cpu(float *input, int n, float *binary) {
    for (int i = 0; i < n; i++) {
        binary[i] = (input[i] > 0) ? 1.0f : -1.0f;
    }
}
 
