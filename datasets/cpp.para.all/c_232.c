#include <stdio.h>
#include <string.h>

// Function declaration
void memcpy_kernel(int *dst, int *src, int n);

int main() {
    // Example data
    const int n = 5;
    int src[] = {1, 2, 3, 4, 5};
    int dst[5];

    // Function call
    memcpy_kernel(dst, src, n * sizeof(int));

    // Output result
    printf("Resultant array after memcpy_kernel:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", dst[i]);
    }

    return 0;
}

// Function definition
void memcpy_kernel(int *dst, int *src, int n) {
    memcpy(dst, src, n);
}
 
