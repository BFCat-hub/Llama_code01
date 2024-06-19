#include <stdio.h>

// Function declaration
void cpuSearchPosShmem1EQ(int key, int *devKey, int *devPos, int size);

int main() {
    // Example data
    const int size = 5;
    int key = 3;
    int devKey[] = {1, 2, 3, 4, 3};
    int devPos[1] = {-1}; // Initialize with an invalid value

    // Function call
    cpuSearchPosShmem1EQ(key, devKey, devPos, size);

    // Output result
    printf("Position of key %d: %d\n", key, devPos[0]);

    return 0;
}

// Function definition
void cpuSearchPosShmem1EQ(int key, int *devKey, int *devPos, int size) {
    for (int globalTx = 0; globalTx < size; globalTx++) {
        if (devKey[globalTx] == key) {
            devPos[0] = globalTx;
        }
    }
}
 
