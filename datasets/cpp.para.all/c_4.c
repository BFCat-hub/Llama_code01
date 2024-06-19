#include <stdio.h>

void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    // 示例用法
    int numElements = 5;
    float x[] = {1.1, 2.2, 3.3, 4.4, 5.5};
    float y[] = {2.0, 4.0, 6.0, 8.0, 10.0};

    printf("原始数组 x：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", x[i]);
    }

    printf("\n原始数组 y：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", y[i]);
    }

    // 调用函数
    add(numElements, x, y);

    printf("\n相加后的数组 y：");
    for (int i = 0; i < numElements; i++) {
        printf("%.2f ", y[i]);
    }

    return 0;
}
