#include <stdio.h>

// Function prototype
void cpuDecodeBitstream(unsigned short *encoded, unsigned short *decoded, int size);

int main() {
    // Example data
    int size = 5;
    unsigned short encoded[] = {1, 0, 0, 1, 1, 1, 1, 0, 1, 0};
    unsigned short decoded[size];

    // Call the function
    cpuDecodeBitstream(encoded, decoded, size);

    // Display the results
    printf("Decoded Bitstream:\n");
    for (int i = 0; i < size; i++) {
        printf("%hu ", decoded[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void cpuDecodeBitstream(unsigned short *encoded, unsigned short *decoded, int size) {
    for (int i = 0; i < size; i++) {
        int bit_index = (i * 2) + 2;
        unsigned short curr_bit = encoded[bit_index];
        decoded[bit_index] = !encoded[bit_index - 1] ^ curr_bit;
        decoded[bit_index + 1] = curr_bit ^ encoded[bit_index + 1];
    }
}
 
