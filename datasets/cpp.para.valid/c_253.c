#include <stdio.h>

// Function declaration
void multMat_cpu(int n, int *arrForce_d, int *arrDistance_d, int *arrAnswer_d);

int main() {
    // Example data
    const int n = 5;
    int arrForce_d[] = {1, 2, 3, 4, 5};
    int arrDistance_d[] = {2, 4, 6, 8, 10};
    int arrAnswer_d[5];

    // Function call
    multMat_cpu(n, arrForce_d, arrDistance_d, arrAnswer_d);

    // Output result
    printf("Resultant array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arrAnswer_d[i]);
    }

    return 0;
}

// Function definition
void multMat_cpu(int n, int *arrForce_d, int *arrDistance_d, int *arrAnswer_d) {
    for (int i = 0; i < n; i++) {
        arrAnswer_d[i] = arrForce_d[i] * arrDistance_d[i];
    }
}
 
