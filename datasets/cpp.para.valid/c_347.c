#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void primal_descent(float *y1, float *y2, float *xbar, float sigma, int w, int h, int nc) {
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            int i;
            float x1, x2, val, norm;
            for (int z = 0; z < nc; z++) {
                i = x + w * y + w * h * z;
                val = xbar[i];
                x1 = (x + 1 < w) ? (xbar[(x + 1) + w * y + w * h * z] - val) : 0.f;
                x2 = (y + 1 < h) ? (xbar[x + w * (y + 1) + w * h * z] - val) : 0.f;
                x1 = y1[i] + sigma * x1;
                x2 = y2[i] + sigma * x2;
                norm = sqrtf(x1 * x1 + x2 * x2);
                y1[i] = x1 / fmax(1.f, norm);
                y2[i] = x2 / fmax(1.f, norm);
            }
        }
    }
}

int main() {
    // Test primal_descent function with a simple example
    int w = 3;
    int h = 3;
    int nc = 2;
    float sigma = 0.1;

    float y1[w * h * nc];
    float y2[w * h * nc];
    float xbar[w * h * nc];

    // Initialize input data (you can modify this part based on your needs)
    for (int i = 0; i < w * h * nc; i++) {
        y1[i] = 1.0;
        y2[i] = 2.0;
        xbar[i] = 3.0;
    }

    // Call the primal_descent function
    primal_descent(y1, y2, xbar, sigma, w, h, nc);

    // Display the results
    printf("Results after primal_descent function:\n");
    for (int i = 0; i < w * h * nc; i++) {
        printf("y1[%d]: %.2f, y2[%d]: %.2f\n", i, y1[i], i, y2[i]);
    }

    return 0;
}
 
