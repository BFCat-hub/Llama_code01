#include <stdio.h>

void initialArray0_cpu(int tasks, int *f3) {
    for (int i = 0; i < tasks; i++) {
        f3[i] = 0;
    }
}

int main() {
    // 示例用法
    int numTasks = 8;
    int array[numTasks];

    printf("原始数组：");
    for (int i = 0; i < numTasks; i++) {
        printf("%d ", array[i]);
    }

    // 调用函数
    initialArray0_cpu(numTasks, array);

    printf("\n初始化后的数组：");
    for (int i = 0; i < numTasks; i++) {
        printf("%d ", array[i]);
    }

    return 0;
}
