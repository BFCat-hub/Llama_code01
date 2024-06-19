#include <stdio.h>

void resetHeap_cpu(int *heap, int *heapPtr, int numBlock) {
    for (int index = 0; index < numBlock; index++) {
        if (index == 0)
            heapPtr[0] = numBlock - 1;
        heap[index] = numBlock - index - 1;
    }
}

int main() {
    // 示例用法
    int numBlocks = 4;
    int heapArray[numBlocks];
    int heapPtrArray[] = {0};

    printf("初始的 heap 数组：");
    for (int i = 0; i < numBlocks; i++) {
        printf("%d ", heapArray[i]);
    }

    printf("\n初始的 heapPtr 数组：%d", heapPtrArray[0]);

    // 调用函数
    resetHeap_cpu(heapArray, heapPtrArray, numBlocks);

    printf("\n重置后的 heap 数组：");
    for (int i = 0; i < numBlocks; i++) {
        printf("%d ", heapArray[i]);
    }

    printf("\n重置后的 heapPtr 数组：%d", heapPtrArray[0]);

    return 0;
}
