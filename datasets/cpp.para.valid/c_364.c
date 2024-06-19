#include <stdio.h>

void nlf_left_forward_cpu(const int n, const float *filters, const int channel, const int height, const int width, const int wsize, float *top_data);

int main() {
    // 示例输入数据，你可以根据实际情况修改
    int n = 1;
    int channel = 1;
    int height = 3;
    int width = 3;
    int wsize = 5;

    float filters[] = {
        // 填充你的滤波器值
    };

    float top_data[n * height * width];  // 初始化 top_data 数组

    // 调用函数
    nlf_left_forward_cpu(n, filters, channel, height, width, wsize, top_data);

    // 打印结果，这里仅为示例，请根据实际情况修改
    printf("Results:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                printf("%f ", top_data[i * height * width + j * width + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}

void nlf_left_forward_cpu(const int n, const float *filters, const int channel, const int height, const int width, const int wsize, float *top_data) {
    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index * step;
        int fbase = index / channel * wsize * step;

        for (int col = width - 1; col >= 0; col--) {
            for (int row = height - 1; row >= 0; row--) {
                float temp = 0;
                int r, c, shift;

                r = row;
                c = col;
                shift = 0 * step + row * width + col;
                temp += top_data[base + r * width + c] * filters[fbase + shift];

                r = row;
                c = col + 1;
                shift = 1 * step + row * width + col;
                if (c < width)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row - 1;
                c = col + 1;
                shift = 2 * step + row * width + col;
                if (c < width && r >= 0)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row + 1;
                c = col + 1;
                shift = 3 * step + row * width + col;
                if (c < width && r < height)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row + 1;
                c = col;
                shift = 4 * step + row * width + col;
                if (r < height)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                top_data[base + row * width + col] = temp;
            }
        }
    }
}
 
