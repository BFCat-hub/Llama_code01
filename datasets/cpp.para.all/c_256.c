#include <stdio.h>

// Function declaration
void zero_centroid_vals_cpu(int k, double *Cx_sum, double *Cy_sum, int *Csize);

int main() {
    // Example data
    const int k = 3;
    double Cx_sum[k], Cy_sum[k];
    int Csize[k];

    // Function call
    zero_centroid_vals_cpu(k, Cx_sum, Cy_sum, Csize);

    // Output result
    printf("Resultant arrays after zeroing centroid values:\n");
    printf("Cx_sum: ");
    for (int i = 0; i < k; i++) {
        printf("%f ", Cx_sum[i]);
    }
    printf("\nCy_sum: ");
    for (int i = 0; i < k; i++) {
        printf("%f ", Cy_sum[i]);
    }
    printf("\nCsize: ");
    for (int i = 0; i < k; i++) {
        printf("%d ", Csize[i]);
    }

    return 0;
}

// Function definition
void zero_centroid_vals_cpu(int k, double *Cx_sum, double *Cy_sum, int *Csize) {
    for (int index = 0; index < k; index++) {
        Cx_sum[index] = 0;
        Cy_sum[index] = 0;
        Csize[index] = 0;
    }
}
 
