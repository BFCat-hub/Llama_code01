#include <stdio.h>

void boundaryCorrectIndexes_cpu(int *d_in, int *d_out, int length, int N) {
    for (int idx = 0; idx < length; idx++) {
        if (d_in[idx] > N) {
            d_out[idx] = N;
        } else {
            d_out[idx] = d_in[idx];
        }
    }
}

int main() {
    // 示例用法
    int arraySize = 5;
    int inputArray[] = {2, 8, 5, 12, 6};
    int outputArray[arraySize];
    int boundaryValue = 10;

    printf("输入数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", inputArray[i]);
    }

    // 调用函数
    boundaryCorrectIndexes_cpu(inputArray, outputArray, arraySize, boundaryValue);

    printf("\n处理后的数组：");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", outputArray[i]);
    }

    return 0;
}
