#include <stdio.h>

void nlf_down_forward_cpu(const int n, const float *filters, const int channel, const int height, const int width, const int wsize, float *top_data);

int main() {
    // Example parameters
    const int n = 2;
    const int channel = 3;
    const int height = 4;
    const int width = 4;
    const int wsize = 5;

    // Example input data (replace with your actual data)
    float filters[n / channel * wsize * height * width];
    float top_data[n * height * width];

    // Initialize your input data here (replace with your actual data)
    for (int i = 0; i < n / channel * wsize * height * width; ++i) {
        filters[i] = i + 1;
    }

    for (int i = 0; i < n * height * width; ++i) {
        top_data[i] = i + 1;
    }

    // Call nlf_down_forward_cpu function
    nlf_down_forward_cpu(n, filters, channel, height, width, wsize, top_data);

    // Print the results or add further processing as needed
    for (int i = 0; i < n * height * width; ++i) {
        printf("%f ", top_data[i]);
    }

    return 0;
}

void nlf_down_forward_cpu(const int n, const float *filters, const int channel, const int height, const int width, const int wsize, float *top_data) {
    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index * step;
        int fbase = index / channel * wsize * step;

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                float temp = 0;
                int r, c, shift;

                r = row;
                c = col;
                shift = 0 * step + row * width + col;
                temp += top_data[base + r * width + c] * filters[fbase + shift];

                r = row - 1;
                c = col;
                shift = 1 * step + row * width + col;
                if (r >= 0) temp += top_data[base + r * width + c] * filters[fbase + shift];
                else temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row - 1;
                c = col - 1;
                shift = 2 * step + row * width + col;
                if (r >= 0 && c >= 0) temp += top_data[base + r * width + c] * filters[fbase + shift];
                else temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row - 1;
                c = col + 1;
                shift = 3 * step + row * width + col;
                if (r >= 0 && c < width) temp += top_data[base + r * width + c] * filters[fbase + shift];
                else temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row;
                c = col - 1;
                shift = 4 * step + row * width + col;
                if (c >= 0) temp += top_data[base + r * width + c] * filters[fbase + shift];
                else temp += top_data[base + row * width + col] * filters[fbase + shift];

                top_data[base + row * width + col] = temp;
            }
        }
    }
}
