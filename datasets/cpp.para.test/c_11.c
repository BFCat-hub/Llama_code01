#include <stdio.h>

void set_sorting_offset(const int nrows, const int ncols, int *offsets) {
    int tid;
    for (tid = 0; tid <= ncols; tid++) {
        offsets[tid] = tid * nrows;
    }
}

int main() {
    // 示例用法
    int numRows = 3;
    int numCols = 4;
    int offsetArray[numCols + 1];

    printf("原始偏移数组：");
    for (int i = 0; i <= numCols; i++) {
        printf("%d ", offsetArray[i]);
    }

    // 调用函数
    set_sorting_offset(numRows, numCols, offsetArray);

    printf("\n设置后的偏移数组：");
    for (int i = 0; i <= numCols; i++) {
        printf("%d ", offsetArray[i]);
    }

    return 0;
}
