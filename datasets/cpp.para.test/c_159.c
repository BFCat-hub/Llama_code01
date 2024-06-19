#include <stdio.h>

void opL23_cpu(float *vec, float *vec1, long depth, long rows, long cols);

int main() {
    // 在这里可以创建测试用的数据，并调用 opL23_cpu 函数
    // 例如：
    long depth = 2;
    long rows = 3;
    long cols = 4;

    // 假设 vec 和 vec1 是相应大小的数组
    float vec[depth * rows * cols];
    float vec1[depth * rows * cols];

    // 初始化数组（这里只是示例，请根据你的实际需求初始化数据）
    for (long i = 0; i < depth * rows * cols; i++) {
        vec[i] = i + 1;
        vec1[i] = i + 2;
    }

    // 调用函数
    opL23_cpu(vec, vec1, depth, rows, cols);

    // 在这里可以添加打印结果或其他操作
    // 例如：
    for (long i = 0; i < depth * rows * cols; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");

    return 0;
}

void opL23_cpu(float *vec, float *vec1, long depth, long rows, long cols) {
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            for (int z = 0; z < depth; z++) {
                unsigned long long i = z * rows * cols + y * cols + x;
                unsigned long long j = z * rows * cols + y * cols;
                unsigned long size2d = cols;
                unsigned long size3d = depth * rows * cols + rows * cols + cols;

                if (i + cols + 1 >= size3d)
                    return;

                vec[i + cols] = 0.5 * (vec1[i + cols] + vec1[i]);

                if (j + 1 >= size2d)
                    return;

                vec[j] = 0.5 * (vec1[j]);
            }
        }
    }
}
