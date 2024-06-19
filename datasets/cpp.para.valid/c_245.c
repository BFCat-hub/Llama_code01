#include <stdio.h>

// Function declaration
void fill_idx(int N, int *device_input, int *device_output);

int main() {
    // Example data
    const int N = 6;
    int device_input[] = {1, 2, 4, 5, 7, 8};
    int device_output[8] = {0}; // Assuming the size is based on the range of device_input

    // Function call
    fill_idx(N, device_input, device_output);

    // Output result
    printf("Resultant device_output array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", device_output[i]);
    }

    return 0;
}

// Function definition
void fill_idx(int N, int *device_input, int *device_output) {
    int idx;
    for (idx = 0; idx + 1 < N; idx++) {
        if (device_input[idx] + 1 == device_input[idx + 1]) {
            device_output[device_input[idx]] = idx;
        }
    }
}
 
