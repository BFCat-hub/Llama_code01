#include <stdio.h>

void upsweep_scan(int twod, int N, int *output) {
    int twod1 = twod * 2;
    int idx;
    for (idx = 0; idx + twod1 - 1 < N; idx += twod1) {
        output[idx + twod1 - 1] += output[idx + twod - 1];
    }
}

int main() {
    // 示例用法
    int arraySize = 8;
    int outputArray[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int twodValue = 1;

    printf("输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", outputArray[i]);
    }

    // 调用函数
    upsweep_scan(twodValue, arraySize, outputArray);

    printf("\n处理后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", outputArray[i]);
    }

    return 0;
}
