#include <stdio.h>

// 函数声明
void kmeans_set_zero(int *means, int size);

int main() {
    // 示例数据
    int means[] = {1, 2, 3, 4, 5};
    int size = sizeof(means) / sizeof(means[0]);

    // 调用函数
    kmeans_set_zero(means, size);

    // 输出结果
    printf("Array after setting all elements to zero:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", means[i]);
    }

    return 0;
}

// 函数定义
void kmeans_set_zero(int *means, int size) {
    for (int id = 0; id < size; id++) {
        means[id] = 0;
    }
}
 
