#include <stdio.h>

// 函数声明
void addIntValues(int *destination, int *value1, int *value2, unsigned int end);

int main() {
    // 示例数据
    const unsigned int end = 5;
    int value1[] = {1, 2, 3, 4, 5};
    int value2[] = {5, 4, 3, 2, 1};
    int destination[5];

    // 调用函数
    addIntValues(destination, value1, value2, end);

    // 输出结果
    printf("Resultant array after elementwise addition of integers:\n");
    for (unsigned int i = 0; i < end; i++) {
        printf("%d ", destination[i]);
    }

    return 0;
}

// 函数定义
void addIntValues(int *destination, int *value1, int *value2, unsigned int end) {
    for (unsigned int i = 0; i < end; i++) {
        destination[i] = value1[i] + value2[i];
    }
}
 
