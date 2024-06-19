#include <stdio.h>

void matrix_mult(int left_rows, int shared_dimensions, int right_columns, float *left, float *right, float *result) {
    int row, column, cell;
    for (row = 0; row < left_rows; row++) {
        for (column = 0; column < right_columns; column++) {
            result[row * right_columns + column] = 0;
            for (cell = 0; cell < shared_dimensions; cell++) {
                result[row * right_columns + column] += left[row * shared_dimensions + cell] * right[cell * right_columns + column];
            }
        }
    }
}

int main() {
    // 示例数据
    const int left_rows = 2;
    const int shared_dimensions = 3;
    const int right_columns = 4;
    float left[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float right[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
    float result[left_rows * right_columns];

    // 调用 matrix_mult 函数
    matrix_mult(left_rows, shared_dimensions, right_columns, left, right, result);

    // 打印输出结果，这里只是一个例子，实际应用中可以根据需要进行处理
    printf("Resultant array result:\n");
    for (int i = 0; i < left_rows; i++) {
        for (int j = 0; j < right_columns; j++) {
            printf("%f ", result[i * right_columns + j]);
        }
        printf("\n");
    }

    return 0;
}
