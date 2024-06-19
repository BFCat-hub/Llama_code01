#include <stdio.h>

void clearLabel(float *prA, float *prB, unsigned int num_nodes, float base) {
    for (unsigned int id = 0; id < num_nodes; id++) {
        prA[id] = base + prA[id] * 0.85;
        prB[id] = 0;
    }
}

int main() {
    // 示例用法
    unsigned int numNodes = 5;
    float prAArray[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float prBArray[numNodes];
    float baseValue = 0.05;

    printf("prA 数组：");
    for (unsigned int i = 0; i < numNodes; i++) {
        printf("%.2f ", prAArray[i]);
    }

    // 调用函数
    clearLabel(prAArray, prBArray, numNodes, baseValue);

    printf("\n清零后的 prA 数组：");
    for (unsigned int i = 0; i < numNodes; i++) {
        printf("%.2f ", prAArray[i]);
    }

    printf("\n清零后的 prB 数组：");
    for (unsigned int i = 0; i < numNodes; i++) {
        printf("%.2f ", prBArray[i]);
    }

    return 0;
}
