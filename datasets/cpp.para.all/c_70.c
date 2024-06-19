#include <stdio.h>

void is_repeat(int N, int *device_input, int *device_output) {
    for (int idx = 0; idx < N; idx++) {
        device_output[idx] = 0;
        if (idx + 1 < N && device_input[idx] == device_input[idx + 1]) {
            device_output[idx] = 1;
        }
    }
}

int main() {
    // 示例用法
    int size = 8;
    int input[] = {1, 2, 2, 3, 4, 4, 4, 5};
    int output[size];

    printf("输入数组：");
    for (int i = 0; i < size; i++) {
        printf("%d ", input[i]);
    }

    // 调用函数
    is_repeat(size, input, output);

    printf("\n重复位置数组：");
    for (int i = 0; i < size; i++) {
        printf("%d ", output[i]);
    }

    return 0;
}
