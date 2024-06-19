#include <stdio.h>

// Function declaration
void Reverse(int *d_in, int *d_out, int size);

int main() {
    // Example data
    const int size = 5;
    int d_in[] = {1, 2, 3, 4, 5};
    int d_out[5];

    // Function call
    Reverse(d_in, d_out, size);

    // Output result
    printf("Reversed array:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", d_out[i]);
    }

    return 0;
}

// Function definition
void Reverse(int *d_in, int *d_out, int size) {
    for (int i = 0; i < size; i++) {
        d_out[i] = d_in[size - 1 - i];
    }
}
 
