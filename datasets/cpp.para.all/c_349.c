#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ENDCOM

void Ring_cpu_kernel(float *A, float *BP, int *corrAB, float *M, int ring, int c, int h, int w) {
    int ringSize = 2 * ring + 1;
    int ringPatch = ringSize * ringSize;
    int size = h * w;

    #pragma omp parallel for ENDCOM
    for (int y1 = 0; y1 < h; y1++) {
        for (int x1 = 0; x1 < w; x1++) {
            int id = y1 * w + x1;
            int x2 = corrAB[2 * id + 0];
            int y2 = corrAB[2 * id + 1];

            for (int dx = -ring; dx <= ring; dx++) {
                for (int dy = -ring; dy <= ring; dy++) {
                    int pIdx = (dy + ring) * ringSize + (dx + ring);
                    int _x2 = x2 + dx, _y2 = y2 + dy;

                    if (_x2 >= 0 && _x2 < w && _y2 >= 0 && _y2 < h) {
                        for (int dc = 0; dc < c; dc++) {
                            M[(dc * size + y1 * w) * ringPatch + pIdx * w + x1] = BP[dc * size + _y2 * w + _x2];
                        }
                    }
                }
            }
        }
    }

    return;
}

int main() {
    // Test Ring_cpu_kernel function with a simple example
    int ring = 1;
    int c = 3; // Number of channels
    int h = 4; // Height
    int w = 4; // Width
    int size = h * w;
    int ringSize = 2 * ring + 1;
    int ringPatch = ringSize * ringSize;

    float *A = (float *)malloc(c * size * sizeof(float));
    float *BP = (float *)malloc(c * size * sizeof(float));
    int *corrAB = (int *)malloc(2 * size * sizeof(int));
    float *M = (float *)malloc(c * size * ringPatch * sizeof(float));

    // Initialize data (you may modify this part based on your actual data)
    for (int i = 0; i < c * size; i++) {
        A[i] = (float)i;
        BP[i] = (float)(i + c * size);
    }

    for (int i = 0; i < 2 * size; i++) {
        corrAB[i] = i % size;
    }

    // Call the Ring_cpu_kernel function
    Ring_cpu_kernel(A, BP, corrAB, M, ring, c, h, w);

    // Display the results (you may modify this part based on your actual data)
    printf("Results after Ring_cpu_kernel function:\n");
    for (int i = 0; i < c * size * ringPatch; i++) {
        printf("%f ", M[i]);
    }

    free(A);
    free(BP);
    free(corrAB);
    free(M);

    return 0;
}
 
