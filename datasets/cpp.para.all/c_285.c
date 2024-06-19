#include <stdio.h>

// Function prototype
void Avg(float *array_avg, float *array, const int r, const int c);

int main() {
    // Example data
    int r = 3;
    int c = 4;
    float array[] = {1.0, 2.0, 3.0, 4.0,
                     5.0, 6.0, 7.0, 8.0,
                     9.0, 10.0, 11.0, 12.0};
    float array_avg[r];

    // Call the function
    Avg(array_avg, array, r, c);

    // Display the results
    printf("Average Array:\n");
    for (int i = 0; i < r; i++) {
        printf("%.2f ", array_avg[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void Avg(float *array_avg, float *array, const int r, const int c) {
    float sum;
    for (int i = 0; i < r; i++) {
        sum = 0.0;
        for (int j = 0; j < c; j++) {
            sum += array[i * c + j];
        }
        array_avg[i] = sum / c;
    }
}
 
