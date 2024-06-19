#include <stdio.h>

void envejecer_kernel_cpu(int *estado, int *edad, int *pupacion, int *N_mobil, int dia) {
    int N = N_mobil[0];
    
    for (int id = 0; id < N; id++) {
        if (dia < 80 || dia > 320) {
            if (edad[id] > pupacion[id]) {
                edad[id]++;
            }
        } else {
            edad[id]++;
        }
    }
}

int main() {
    // 示例数据
    const int N = 5;
    int estado[N] = {1, 1, 1, 0, 1};
    int edad[N] = {75, 90, 100, 50, 60};
    int pupacion[N] = {70, 80, 95, 45, 55};
    int N_mobil[1] = {N};
    int dia = 100;

    // 调用 envejecer_kernel_cpu 函数
    envejecer_kernel_cpu(estado, edad, pupacion, N_mobil, dia);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Edad after envejecer_kernel_cpu:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", edad[i]);
    }
    printf("\n");

    return 0;
}
