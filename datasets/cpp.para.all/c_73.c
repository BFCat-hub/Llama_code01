#include <stdio.h>

// 定义 max 函数
int max(int a, int b) {
    return (a > b) ? a : b;
}

void compute_new_means(float *mx, float *my, const float *sx, const float *sy, const int *c, int size) {
    int cluster = 0;
    const int count = max(1, c[cluster]);

    for (cluster = 0; cluster < size; cluster++) {
        mx[cluster] = sx[cluster] / count;
        my[cluster] = sy[cluster] / count;
    }
}

int main() {
    // 示例用法
    int size = 3;
    float mx[] = {1.0, 2.0, 3.0};
    float my[] = {4.0, 5.0, 6.0};
    float sx[] = {7.0, 8.0, 9.0};
    float sy[] = {10.0, 11.0, 12.0};
    int c[] = {2, 0, 1};

    printf("输入 mx 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", mx[i]);
    }

    printf("\n输入 my 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", my[i]);
    }

    printf("\n输入 sx 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", sx[i]);
    }

    printf("\n输入 sy 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", sy[i]);
    }

    printf("\n输入 c 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", c[i]);
    }

    // 调用函数
    compute_new_means(mx, my, sx, sy, c, size);

    printf("\n计算新均值后的 mx 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", mx[i]);
    }

    printf("\n计算新均值后的 my 数组：\n");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", my[i]);
    }

    return 0;
}
