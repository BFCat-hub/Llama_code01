#include <stdio.h>

void cpu_laplace_filter(float *Img, float *laplace, float _dz, float _dx, int npml, int nnz, int nnx);

int main() {
    // Example dimensions
    int npml = 2;
    int nnz = 5;
    int nnx = 4;

    // Example input data
    float Img[20] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0
    };

    // Output array for laplace
    float laplace_result[20];

    // Applying cpu_laplace_filter
    cpu_laplace_filter(Img, laplace_result, 0.1, 0.2, npml, nnz, nnx);

    // Print the result
    printf("Input Img:\n");
    for (int i = 0; i < nnz; ++i) {
        for (int j = 0; j < nnx; ++j) {
            printf("%8.4f ", Img[i + j * nnz]);
        }
        printf("\n");
    }

    printf("\nOutput Laplace:\n");
    for (int i = 0; i < nnz; ++i) {
        for (int j = 0; j < nnx; ++j) {
            printf("%8.4f ", laplace_result[i + j * nnz]);
        }
        printf("\n");
    }

    return 0;
}

void cpu_laplace_filter(float *Img, float *laplace, float _dz, float _dx, int npml, int nnz, int nnx) {
    for (int i1 = npml; i1 < nnz - npml; i1++) {
        for (int i2 = npml; i2 < nnx - npml; i2++) {
            int id = i1 + i2 * nnz;
            float diff1 = 0.0f;
            float diff2 = 0.0f;
            diff1 = Img[id + 1] - 2.0 * Img[id] + Img[id - 1];
            diff2 = Img[id + nnz] - 2.0 * Img[id] + Img[id - nnz];
            laplace[id] = _dz * _dz * diff1 + _dx * _dx * diff2;
        }
    }
}
 
