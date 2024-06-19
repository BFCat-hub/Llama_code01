#include <stdio.h>

// 函数声明
void histogram_cpu(int n, int *color, int *bucket);

int main() {
    // 示例数据
    int n = 8;
    int color[] = {1, 2, 1, 3, 2, 3, 1, 2};
    int bucket[4] = {0}; // Assuming 4 buckets

    // 调用函数
    histogram_cpu(n, color, bucket);

    // 输出结果
    printf("Histogram:\n");
    for (int i = 0; i < 4; i++) {
        printf("Bucket %d: %d\n", i, bucket[i]);
    }

    return 0;
}

// 函数定义
void histogram_cpu(int n, int *color, int *bucket) {
    for (int i = 0; i < n; i++) {
        int c = color[i];
        bucket[c] += 1;
    }
}
 
