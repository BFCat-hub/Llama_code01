#include <stdio.h>

// 定义 im2col_get_pixel 函数，这里需要根据实际情况实现
float im2col_get_pixel(const float *data_im, int height, int width, int channels, int row, int col, int channel, int pad) {
    // 这里只是一个示例，实际实现需要根据具体需求来编写
    // 在这里，我们简单返回一个固定的值（实际上应该是从输入数据中获取像素值）
    return 1.0;
}

void im2col_cpu(float *data_im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);

int main() {
    // 在这里可以创建测试用的数据，并调用 im2col_cpu 函数
    // 例如：
    int channels = 3;
    int height = 4;
    int width = 4;
    int ksize = 2;
    int stride = 2;
    int pad = 0;

    // 假设 data_im 和 data_col 是相应大小的数组
    float data_im[channels * height * width];
    float data_col[channels * ksize * ksize * ((height - ksize + 2 * pad) / stride + 1) * ((width - ksize + 2 * pad) / stride + 1)];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (int i = 0; i < channels * height * width; i++) {
        data_im[i] = i + 1;
    }

    // 调用函数
    im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (int i = 0; i < channels * ksize * ksize * ((height - ksize + 2 * pad) / stride + 1) * ((width - ksize + 2 * pad) / stride + 1); i++) {
        printf("data_col[%d] = %f\n", i, data_col[i]);
    }

    return 0;
}

void im2col_cpu(float *data_im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col) {
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;

        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}
