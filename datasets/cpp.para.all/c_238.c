#include <stdio.h>

// Function declaration
void kernelIsFirst_cpu(int *head, int *first_pts, int n);

int main() {
    // Example data
    const int n = 5;
    int head[] = {1, 0, 1, 0, 1};
    int first_pts[5];

    // Function call
    kernelIsFirst_cpu(head, first_pts, n);

    // Output result
    printf("Resultant first_pts array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", first_pts[i]);
    }

    return 0;
}

// Function definition
void kernelIsFirst_cpu(int *head, int *first_pts, int n) {
    for (int i = 0; i < n; i++) {
        if (head[i] == 1) {
            first_pts[i] = i;
        } else {
            first_pts[i] = 0;
        }
    }
}
 
