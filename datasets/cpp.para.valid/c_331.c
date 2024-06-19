#include <stdio.h>

void deInterleave_cpu2(float *d_X_out, float *d_Y_out, char *d_XY_in, int pitch_out, int pitch_in, int width, int height);

int main() {
    // Example dimensions
    int width = 3;
    int height = 2;

    // Example input data (assuming 2 floats per element)
    char input[2][3 * 2 * sizeof(float)] = {
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0, 10.0, 11.0, 12.0}
    };

    // Output arrays for X and Y
    float output_X[2][3];
    float output_Y[2][3];

    // Applying deInterleave_cpu2
    deInterleave_cpu2((float *)output_X, (float *)output_Y, (char *)input, sizeof(float) * 3, sizeof(char) * 2 * 3 * sizeof(float), width, height);

    // Print the result
    printf("Output X:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%8.4f ", output_X[y][x]);
        }
        printf("\n");
    }

    printf("\nOutput Y:\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%8.4f ", output_Y[y][x]);
        }
        printf("\n");
    }

    return 0;
}

void deInterleave_cpu2(float *d_X_out, float *d_Y_out, char *d_XY_in, int pitch_out, int pitch_in, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            float *data = (float *)(d_XY_in + y * pitch_in) + 2 * x;
            *((float *)((char *)d_X_out + y * pitch_out) + x) = data[0];
            *((float *)((char *)d_Y_out + y * pitch_out) + x) = data[1];
        }
    }
}
 
