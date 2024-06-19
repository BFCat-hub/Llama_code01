#include <stdio.h>

void nlf_filter_up_backward_cpu(const int n, const float *bottom_data, const float *top_data, const float *temp_diff, const int channel, const int height, const int width, const int wsize, float *filters_diff);

int main() {
    // 示例输入数据，你可以根据实际情况修改
    int n = 1;
    int channel = 1;
    int height = 3;
    int width = 3;
    int wsize = 5;

    float bottom_data[n * channel * height * width];  // Initialize bottom_data array
    float top_data[n * channel * height * width];     // Initialize top_data array
    float temp_diff[n * channel * height * width];    // Initialize temp_diff array
    float filters_diff[n * channel * wsize * height * width];  // Initialize filters_diff array

    // 调用函数
    nlf_filter_up_backward_cpu(n, bottom_data, top_data, temp_diff, channel, height, width, wsize, filters_diff);

    // 打印结果，这里仅为示例，请根据实际情况修改
    printf("Results:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < channel; ++j) {
            for (int k = 0; k < wsize; ++k) {
                for (int m = 0; m < height; ++m) {
                    for (int n = 0; n < width; ++n) {
                        printf("%f ", filters_diff[i * channel * wsize * height * width + j * wsize * height * width + k * height * width + m * width + n]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
    }

    return 0;
}

void nlf_filter_up_backward_cpu(const int n, const float *bottom_data, const float *top_data, const float *temp_diff, const int channel, const int height, const int width, const int wsize, float *filters_diff) {
    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index / step * step * channel + index % step;
        int fbase = index / step * step * wsize + index % step;
        int row = index % step / width;
        int col = index % step % width;

        for (int i = 0; i < channel; i++) {
            filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height)
                filters_diff[fbase + step] += temp_diff[base + i * step] * top_data[base + width + i * step];
            else
                filters_diff[fbase + step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height && col - 1 >= 0)
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * top_data[base + width - 1 + i * step];
            else
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height && col + 1 < width)
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * top_data[base + width + 1 + i * step];
            else
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col + 1 < width)
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * top_data[base + 1 + i * step];
            else
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];
        }
    }
}
 
