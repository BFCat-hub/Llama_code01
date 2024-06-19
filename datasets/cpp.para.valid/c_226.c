#include <stdio.h>

// 函数声明
void histo_cpu(const unsigned int *const vals, unsigned int *const histo, int numVals);

int main() {
    // 示例数据
    const int numVals = 10;
    unsigned int vals[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    unsigned int histo[6] = {0};

    // 调用函数
    histo_cpu(vals, histo, numVals);

    // 输出结果
    printf("Histogram result:\n");
    for (int i = 0; i < 6; i++) {
        printf("Value %d: %d occurrences\n", i, histo[i]);
    }

    return 0;
}

// 函数定义
void histo_cpu(const unsigned int *const vals, unsigned int *const histo, int numVals) {
    for (int i = 0; i < numVals; i++) {
        histo[vals[i]] = histo[vals[i]] + 1;
    }
}
 
