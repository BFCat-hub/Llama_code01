#include <stdio.h>

// Function declaration
void subtractIntValues(int *destination, int *value1, int *value2, unsigned int end);

int main() {
    // Example data
    const unsigned int end = 5;
    int value1[] = {10, 8, 6, 4, 2};
    int value2[] = {5, 4, 3, 2, 1};
    int destination[5];

    // Function call
    subtractIntValues(destination, value1, value2, end);

    // Output result
    printf("Resultant array after elementwise subtraction of integers:\n");
    for (unsigned int i = 0; i < end; i++) {
        printf("%d ", destination[i]);
    }

    return 0;
}

// Function definition
void subtractIntValues(int *destination, int *value1, int *value2, unsigned int end) {
    for (unsigned int i = 0; i < end; i++) {
        destination[i] = value1[i] - value2[i];
    }
}
 
