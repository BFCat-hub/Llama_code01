#include <stdio.h>

void pad_input(float *f_in, float *f_out, int H, int W, int D, int pad) {
    int col, row, dep;
    int new_H = H + 2 * pad;
    int new_W = W + 2 * pad;

    for (col = 0; col < new_H; col++) {
        for (row = 0; row < new_W; row++) {
            for (dep = 0; dep < D; dep++) {
                int i = dep * new_H * new_W + col * new_W + row;
                int j = dep * H * W + (col - pad) * W + (row - pad);

                if ((col < pad || col > H + pad - 1) || (row < pad || row > W + pad - 1))
                    f_out[i] = 0;
                else
                    f_out[i] = f_in[j];
            }
        }
    }
}

int main() {
    // Test pad_input function with a simple example
    int H = 4;
    int W = 4;
    int D = 2;
    int pad = 1;

    float input[H * W * D];
    float output[(H + 2 * pad) * (W + 2 * pad) * D];

    // Initialize input array
    for (int i = 0; i < H * W * D; i++) {
        input[i] = i + 1;  // Just an example, you can modify this based on your needs
    }

    printf("Input:\n");
    for (int i = 0; i < H * W * D; i++) {
        printf("%.2f ", input[i]);
    }
    printf("\n");

    // Call pad_input function
    pad_input(input, output, H, W, D, pad);

    printf("Output:\n");
    for (int i = 0; i < (H + 2 * pad) * (W + 2 * pad) * D; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

    return 0;
}
 
