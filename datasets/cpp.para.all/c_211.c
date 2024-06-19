#include <stdio.h>

// 函数声明
void addV_cpu(int *a, int *b, int *c, int N);

int main() {
    // 示例数据
    int N = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {5, 4, 3, 2, 1};
    int c[5];

    // 调用函数
    addV_cpu(a, b, c, N);

    // 输出结果
    printf("Resultant array after addition:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", c[i]);
    }

    return 0;
}

// 函数定义
void addV_cpu(int *a, int *b, int *c, int N) {
    for (int index = 0; index < N; index++) {
        c[index] = a[index] + b[index];
    }
}
 
