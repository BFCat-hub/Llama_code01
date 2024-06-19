#include <stdio.h>

// Function declaration
void find_max_cpu(int *data, int N);

int main() {
    // Example data
    const int N = 5;
    int data[] = {8, 3, 12, 5, 7};

    // Function call
    find_max_cpu(data, N);

    // Output result
    printf("Maximum value in the array: %d\n", data[0]);

    return 0;
}

// Function definition
void find_max_cpu(int *data, int N) {
    int m = data[0];

    for (int i = 0; i < N; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }

    data[0] = m;
}
 
