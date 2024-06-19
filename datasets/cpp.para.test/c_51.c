#include <stdio.h>

void delay_kernel_cpu(int *N_mobil, int *Tau, int dia) {
    int N = N_mobil[0];
    for (int id = 0; id < N; id++) {
        if (Tau[id] > 0) {
            Tau[id] = Tau[id] - 1;
        }
    }
}

int main() {
    // 示例用法
    int numElements = 5;
    int N_mobilArray[] = {numElements};
    int TauArray[] = {2, 0, 4, 1, 0};
    int diaValue = 0;

    printf("Tau 数组（初始）：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", TauArray[i]);
    }

    // 调用函数
    delay_kernel_cpu(N_mobilArray, TauArray, diaValue);

    printf("\n延迟后的 Tau 数组：");
    for (int i = 0; i < numElements; i++) {
        printf("%d ", TauArray[i]);
    }

    return 0;
}
