#include <stdio.h>

// Function prototype
void check_results_kernel(unsigned int *g_results0, unsigned int *g_results1, int n);

int main() {
    // Example data
    int n = 5;
    unsigned int g_results0[] = {1, 2, 3, 4, 5};
    unsigned int g_results1[] = {1, 2, 8, 4, 5};  // Intentionally introduced difference at index 2

    // Call the function
    check_results_kernel(g_results0, g_results1, n);

    return 0;
}

// Function definition
void check_results_kernel(unsigned int *g_results0, unsigned int *g_results1, int n) {
    unsigned int gidx;
    unsigned int result0;
    unsigned int result1;
    for (gidx = 0; gidx < n; gidx++) {
        result0 = g_results0[gidx];
        result1 = g_results1[gidx];
        if (result0 != result1) {
            printf("%u != %u for %u\n", result0, result1, gidx);
        }
    }
}
 
