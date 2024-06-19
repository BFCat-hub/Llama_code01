#include <stdio.h>

void nlf_filter_left_backward_cpu(const int n, const float *bottom_data, const float *top_data, const float *temp_diff, const int channel, const int height, const int width, const int wsize, float *filters_diff);

int main() {
    // Example parameters
    const int n = 2;
    const int channel = 3;
    const int height = 4;
    const int width = 4;
    const int wsize = 5;

    // Example input data (replace with your actual data)
    float bottom_data[n * channel * height * width];
    float top_data[n * channel * height * width];
    float temp_diff[n * channel * height * width];
    float filters_diff[n / height * height * wsize];

    // Initialize your input data here (replace with your actual data)
    for (int i = 0; i < n * channel * height * width; ++i) {
        bottom_data[i] = i + 1;
        top_data[i] = i + 1;
        temp_diff[i] = i + 1;
    }

    for (int i = 0; i < n / height * height * wsize; ++i) {
        filters_diff[i] = i + 1;
    }

    // Call nlf_filter_left_backward_cpu function
    nlf_filter_left_backward_cpu(n, bottom_data, top_data, temp_diff, channel, height, width, wsize, filters_diff);

    // Print the results or add further processing as needed
    for (int i = 0; i < n / height * height * wsize; ++i) {
        printf("%f ", filters_diff[i]);
    }

    return 0;
}

void nlf_filter_left_backward_cpu(const int n, const float *bottom_data, const float *top_data, const float *temp_diff, const int channel, const int height, const int width, const int wsize, float *filters_diff) {
    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index / step * step * channel + index % step;
        int fbase = index / step * step * wsize + index % step;
        int row = index % step / width;
        int col = index % step % width;

        for (int i = 0; i < channel; i++) {
            filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col + 1 < width)
                filters_diff[fbase + step] += temp_diff[base + i * step] * top_data[base + 1 + i * step];
            else
                filters_diff[fbase + step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col + 1 < width && row - 1 >= 0)
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * top_data[base - width + 1 + i * step];
            else
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col + 1 < width && row + 1 < height)
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * top_data[base + width + 1 + i * step];
            else
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height)
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * top_data[base + width + i * step];
            else
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];
        }
    }
}
