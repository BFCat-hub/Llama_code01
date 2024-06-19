#include <stdio.h>

// Function declaration
void transferMBR3_cpu(double *xy_copy, long long *a_copy, int tasks);

int main() {
    // Example data
    const int tasks = 5;
    double xy_copy[] = {1.5, 2.5, 3.5, 4.5, 5.5};
    long long a_copy[5];

    // Function call
    transferMBR3_cpu(xy_copy, a_copy, tasks);

    // Output result
    printf("Resultant array after transferMBR3_cpu:\n");
    for (int i = 0; i < tasks; i++) {
        printf("%lld ", a_copy[i]);
    }

    return 0;
}

// Function definition
void transferMBR3_cpu(double *xy_copy, long long *a_copy, int tasks) {
    for (int i = 0; i < tasks; i++) {
        a_copy[i] = xy_copy[i] * 10000000;
    }
}
 
