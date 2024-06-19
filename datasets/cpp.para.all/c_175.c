#include <stdio.h>
#include <assert.h>

void eltwise_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out, int sum, int mult);

int main() {
    // Example parameters
    const int batch = 2;
    const int w1 = 4, h1 = 3, c1 = 2;
    const int w2 = 2, h2 = 3, c2 = 1;
    const int sum = 1, mult = 0;

    // Example input data (replace with your actual data)
    float add[batch * w1 * h1 * c1];
    float out[w2 * h2 * c2 * batch];

    // Initialize your input data here (replace with your actual data)
    for (int i = 0; i < batch * w1 * h1 * c1; ++i) {
        add[i] = i + 1;
    }

    // Call eltwise_cpu function
    eltwise_cpu(batch, w1, h1, c1, add, w2, h2, c2, out, sum, mult);

    // Print the results or add further processing as needed
    for (int i = 0; i < w2 * h2 * c2 * batch; ++i) {
        printf("%f ", out[i]);
    }

    return 0;
}

void eltwise_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out, int sum, int mult) {
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);

    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;

    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i, j, k, b;

    if (mult == 1) {
        for (b = 0; b < batch; ++b) {
            for (k = 0; k < minc; ++k) {
                for (j = 0; j < minh; ++j) {
                    for (i = 0; i < minw; ++i) {
                        int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                        int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                        out[out_index] = out[out_index] * add[add_index];
                    }
                }
            }
        }
    } else if (sum == 1) {
        for (b = 0; b < batch; ++b) {
            for (k = 0; k < minc; ++k) {
                for (j = 0; j < minh; ++j) {
                    for (i = 0; i < minw; ++i) {
                        int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                        int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                        out[out_index] = out[out_index] + add[add_index];
                    }
                }
            }
        }
    }
}
