#include <stdio.h>
#include <math.h>

// Function prototype
void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c, int size);

int main() {
    // Example data
    int size = 5;
    unsigned char a[] = {3, 4, 5, 6, 7};
    unsigned char b[] = {4, 5, 6, 7, 8};
    unsigned char c[size];

    // Call the function
    pythagoras(a, b, c, size);

    // Display the result
    printf("Resultant C:\n");
    for (int i = 0; i < size; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c, int size) {
    for (int idx = 0; idx < size; idx++) {
        float af = (float)(a[idx]);
        float bf = (float)(b[idx]);
        c[idx] = (unsigned char)sqrtf(af * af + bf * bf);
    }
}
 
