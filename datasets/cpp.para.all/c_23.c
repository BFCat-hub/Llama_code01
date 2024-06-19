#include <stdio.h>

void getCanBusData(int *canData, int size) {
    for (int idx = 0; idx < size; idx++) {
        canData[idx] += 1;
    }
}

int main() {
    // 示例用法
    int dataSize = 8;
    int canData[dataSize] = {10, 20, 30, 40, 50, 60, 70, 80};

    printf("原始 CAN 数据：");
    for (int i = 0; i < dataSize; i++) {
        printf("%d ", canData[i]);
    }

    // 调用函数
    getCanBusData(canData, dataSize);

    printf("\n处理后的 CAN 数据：");
    for (int i = 0; i < dataSize; i++) {
        printf("%d ", canData[i]);
    }

    return 0;
}
#include <stdio.h>

void getCanBusData(int *canData, int size) {
    for (int idx = 0; idx < size; idx++) {
        canData[idx] += 1;
    }
}

int main() {
    // 示例用法
    int dataSize = 8;
    int canData[dataSize] = {10, 20, 30, 40, 50, 60, 70, 80};

    printf("原始 CAN 数据：");
    for (int i = 0; i < dataSize; i++) {
        printf("%d ", canData[i]);
    }

    // 调用函数
    getCanBusData(canData, dataSize);

    printf("\n处理后的 CAN 数据：");
    for (int i = 0; i < dataSize; i++) {
        printf("%d ", canData[i]);
    }

    return 0;
}
