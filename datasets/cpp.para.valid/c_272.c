#include <stdio.h>

// Function prototype
float reduceCPU(float *data, int size);

int main() {
    // Example data
    int size = 5;
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Call the function
    float result = reduceCPU(data, size);

    // Display the result
    printf("Sum: %.2f\n", result);

    return 0;
}

// Function definition
float reduceCPU(float *data, int size) {
    float sum = data[0];
    float c = (float)0.0;

    for (int i = 1; i < size; i++) {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}
 
