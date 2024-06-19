#include <stdio.h>

// Function prototype
void matrixTranspose_cpu(int *in_mat, int *out_mat, int dim_rows, int dim_cols);

int main() {
    // Example data
    int dim_rows = 3;
    int dim_cols = 4;
    int in_mat[] = {1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12};
    int out_mat[dim_cols * dim_rows];

    // Call the function
    matrixTranspose_cpu(in_mat, out_mat, dim_rows, dim_cols);

    // Display the results
    printf("Transposed Matrix:\n");
    for (int i = 0; i < dim_cols; i++) {
        for (int j = 0; j < dim_rows; j++) {
            printf("%d ", out_mat[i * dim_rows + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void matrixTranspose_cpu(int *in_mat, int *out_mat, int dim_rows, int dim_cols) {
    for (int i = 0; i < dim_rows; ++i) {
        for (int j = 0; j < dim_cols; ++j) {
            unsigned int new_pos = j * dim_rows + i;
            out_mat[new_pos] = in_mat[i * dim_cols + j];
        }
    }
}
 
