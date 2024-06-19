#include <stdio.h>

void testInt1_cpu(const int *input, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        int sum = 0;
        for (int i = 0; i < 3000 * 4; i++) {
            if (input[i] == 0) {
                sum++;
            }
        }
    }
}

int main() {
    // 示例用法
    int arraySize = 3000 * 4;
    int inputArray[arraySize];

    // 假设 inputArray 初始化为一些值，这里简化为都设为 0
    for (int i = 0; i < arraySize; i++) {
        inputArray[i] = 0;
    }

    // 调用函数
    testInt1_cpu(inputArray, arraySize);

    return 0;
}
