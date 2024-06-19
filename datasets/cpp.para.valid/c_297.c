#include <stdio.h>

// Function prototype
void Argmax(int *argMax, float **array, const int r, const int c);

int main() {
    // Example data
    int r = 3;
    int c = 4;
    float data[3][4] = {{1.2, 2.3, 0.5, 1.8},
                        {0.9, 2.0, 1.5, 3.2},
                        {2.1, 1.4, 3.0, 0.8}};
    float *array[r];
    for (int i = 0; i < r; i++) {
        array[i] = data[i];
    }
    int argMax[r];

    // Call the function
    Argmax(argMax, array, r, c);

    // Display the results
    printf("Argmax Indices:\n");
    for (int i = 0; i < r; i++) {
        printf("%d ", argMax[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void Argmax(int *argMax, float **array, const int r, const int c) {
    int idx;
    float temp;
    for (int i = 0; i < r; i++) {
        idx = 0;
        temp = 0.0;
        for (int j = 0; j < c; j++) {
            if (array[i][j] > temp) {
                temp = array[i][j];
                idx = j;
            }
        }
        argMax[i] = idx;
    }
}
 
