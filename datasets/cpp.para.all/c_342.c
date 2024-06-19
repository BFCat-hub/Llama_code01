#include <stdio.h>

void opL21_cpu(float *vec, float *vec1, long depth, long rows, long cols) {
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long long i = z * rows * cols + y * cols + x;
                unsigned long long j = z * rows * cols + x;
                unsigned long size2d = cols;
                unsigned long size3d = depth * rows * cols + rows * cols + cols;
                if (i + cols + 1 >= size3d || j + 1 >= size2d) {
                    return;
                }
                vec[i + cols] = 0.25 * (vec1[i + 1] + vec1[i] + vec1[i + cols + 1] + vec1[i + cols]);
                vec[j] = (vec1[j] + vec1[j + 1]) / 4;
            }
        }
    }
}

int main() {
    // Test opL21_cpu function with a simple example
    long depth = 3;
    long rows = 4;
    long cols = 5;

    float vec1[depth * rows * cols];
    float vec[depth * rows * cols];

    // Initialize input vec1 with some values
    for (long i = 0; i < depth * rows * cols; i++) {
        vec1[i] = i + 1;
    }

    opL21_cpu(vec, vec1, depth, rows, cols);

    // Display the output vec after applying opL21_cpu
    printf("Output vec:\n");
    for (long i = 0; i < depth * rows * cols; i++) {
        printf("%.2f ", vec[i]);
    }
    printf("\n");

    return 0;
}
 
