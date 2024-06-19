#include <stdio.h>

// Function prototype
void convertInstanceToLabel_Kernel_cpu(unsigned short *d_outputLabel, const unsigned char *d_inputInstance, const unsigned short *d_instanceToLabel, unsigned int width, unsigned int height);

int main() {
    // Example data
    unsigned int width = 3;
    unsigned int height = 3;
    unsigned char d_inputInstance[] = {1, 2, 3,
                                        1, 2, 3,
                                        1, 2, 3};
    unsigned short d_instanceToLabel[] = {10, 20, 30};
    unsigned short d_outputLabel[width * height];

    // Call the function
    convertInstanceToLabel_Kernel_cpu(d_outputLabel, d_inputInstance, d_instanceToLabel, width, height);

    // Display the results
    printf("Converted Labels:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%hu ", d_outputLabel[y * width + x]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void convertInstanceToLabel_Kernel_cpu(unsigned short *d_outputLabel, const unsigned char *d_inputInstance, const unsigned short *d_instanceToLabel, unsigned int width, unsigned int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            d_outputLabel[y * width + x] = d_instanceToLabel[d_inputInstance[y * width + x]];
        }
    }
}
 
