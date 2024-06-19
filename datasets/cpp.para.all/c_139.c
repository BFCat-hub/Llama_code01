#include <stdio.h>

void grad_x_cpu(const float *u, float *grad, long depth, long rows, long cols) {
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long size2d = rows * cols;
                unsigned long long idx = z * size2d + y * cols + x;
                float uidx = u[idx];

                if (x > 0) {
                    grad[idx] = (uidx - u[z * size2d + y * cols + (x - 1)]);
                }
            }
        }
    }
}

int main() {
    // Example usage
    long depth = 3;
    long rows = 4;
    long cols = 5;

    // Assuming your 3D array is a flat 1D array
    float u[depth * rows * cols];

    // Initialize your array with some values (for example, 1.0 for simplicity)
    for (int i = 0; i < depth * rows * cols; i++) {
        u[i] = 1.0f;
    }

    // Allocate memory for the gradient array
    float grad[depth * rows * cols];

    // Call the function to compute the gradient
    grad_x_cpu(u, grad, depth, rows, cols);

    // Print the result (for demonstration purposes)
    for (int i = 0; i < depth * rows * cols; i++) {
        printf("%f ", grad[i]);
    }

    return 0;
}
