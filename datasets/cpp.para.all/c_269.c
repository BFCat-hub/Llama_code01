#include <stdio.h>

// Function prototype
void cpuSearchPosShmem1(int key, int *gpu_key_arr, int *gpu_pos, int size);

int main() {
    // Example data
    int size = 5;
    int key = 7;
    int gpu_key_arr[] = {3, 6, 8, 10, 12};
    int gpu_pos;

    // Call the function
    cpuSearchPosShmem1(key, gpu_key_arr, &gpu_pos, size);

    // Display the result
    printf("Position of key %d: %d\n", key, gpu_pos);

    return 0;
}

// Function definition
void cpuSearchPosShmem1(int key, int *gpu_key_arr, int *gpu_pos, int size) {
    for (int globalTx = 0; globalTx < size - 1; globalTx++) {
        if (key >= gpu_key_arr[globalTx] && key < gpu_key_arr[globalTx + 1]) {
            *gpu_pos = globalTx;
            return; // Assuming that the key can only exist in one position
        }
    }

    // If the key is greater than or equal to the last element, set the position to the last index
    if (key >= gpu_key_arr[size - 1]) {
        *gpu_pos = size - 1;
    }
}
 
