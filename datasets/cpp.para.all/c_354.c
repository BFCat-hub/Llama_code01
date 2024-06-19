#include <stdio.h>
#include <stdlib.h>

void opLadj2_cpu(float *vec, float *vec1, float *vec2, float *vec3, long depth, long rows, long cols) {
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long long i = z * rows * cols + y * cols + x;
                unsigned long long j = z * rows * cols + y * cols;
                unsigned long size2d = z * rows * cols + cols * rows;
                unsigned long size3d = depth * rows * cols + rows * cols + cols;

                if (i + cols + 1 >= size3d) return;

                vec[i + 1] = vec1[i + 1] + 0.25 * (vec2[i + 1] + vec2[i] + vec2[i + cols + 1] + vec2[i + cols]) + 0.5 * (vec3[i + 1] + vec3[i + cols + 1]);

                if (j + cols >= size2d) return;

                vec[j] = vec1[j] + (vec2[j] + vec2[j + cols]) / 4 + (vec3[j] + vec3[j + cols]) / 2;
            }
        }
    }
}

int main() {
    // Test opLadj2_cpu function with a simple example
    int depth = 3;
    int rows = 3;
    int cols = 3;

    float *vec = (float *)malloc(depth * rows * cols * sizeof(float));
    float *vec1 = (float *)malloc(depth * rows * cols * sizeof(float));
    float *vec2 = (float *)malloc(depth * rows * cols * sizeof(float));
    float *vec3 = (float *)malloc(depth * rows * cols * sizeof(float));

    // Initialize data (you may modify this part based on your actual data)
    for (int i = 0; i < depth * rows * cols; i++) {
        vec1[i] = (float)i;
        vec2[i] = (float)i * 2;
        vec3[i] = (float)i * 3;
    }

    // Call the opLadj2_cpu function
    opLadj2_cpu(vec, vec1, vec2, vec3, depth, rows, cols);

    // Display the results (you may modify this part based on your actual data)
    printf("Results after opLadj2_cpu function:\n");
    for (int i = 0; i < depth * rows * cols; i++) {
        printf("%f ", vec[i]);
    }

    free(vec);
    free(vec1);
    free(vec2);
    free(vec3);

    return 0;
}
 
