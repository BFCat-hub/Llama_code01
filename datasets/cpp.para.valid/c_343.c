#include <stdio.h>

void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out) {
    int b, i, j, k;
    int out_c = c / (stride * stride);

    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h; ++j) {
                for (i = 0; i < w; ++i) {
                    int in_index = i + w * (j + h * (k + c * b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i * stride + offset % stride;
                    int h2 = j * stride + offset / stride;
                    int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));

                    if (forward)
                        out[out_index] = x[in_index];
                    else
                        out[in_index] = x[out_index];
                }
            }
        }
    }
}

int main() {
    // Test reorg_cpu function with a simple example
    int w = 4;
    int h = 4;
    int c = 3;
    int batch = 2;
    int stride = 2;

    float x[w * h * c * batch];
    float out[w * h * c * batch];

    // Initialize input x with some values
    for (int i = 0; i < w * h * c * batch; i++) {
        x[i] = i + 1;
    }

    reorg_cpu(x, w, h, c, batch, stride, 1, out);  // Forward pass

    // Display the output after applying reorg_cpu with forward pass
    printf("Output (Forward Pass):\n");
    for (int i = 0; i < w * h * c * batch; i++) {
        printf("%.2f ", out[i]);
    }
    printf("\n");

    reorg_cpu(x, w, h, c, batch, stride, 0, out);  // Backward pass

    // Display the output after applying reorg_cpu with backward pass
    printf("Output (Backward Pass):\n");
    for (int i = 0; i < w * h * c * batch; i++) {
        printf("%.2f ", out[i]);
    }
    printf("\n");

    return 0;
}
 
