#include <stdio.h>

// Function prototype
void avgpool_cpu(int n, float *input_im, float *output_im);

int main() {
    // Example data
    int n = 3;
    float input_im[] = {1.0, 2.0, 3.0, /* ... */ 9.0, 10.0, 11.0, /* ... */ 17.0, 18.0, 19.0};
    float output_im[n];

    // Call the function
    avgpool_cpu(n, input_im, output_im);

    // Display the results
    printf("Output Image:\n");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", output_im[i]);
    }
    printf("\n");

    return 0;
}

// Function definition
void avgpool_cpu(int n, float *input_im, float *output_im) {
    for (int class_index = 0; class_index < n; class_index++) {
        float *tmp_input = input_im + 169 * class_index;
        float tmp = 0.0f;

        for (int i = 0; i < 169; i++) {
            tmp += tmp_input[i];
        }

        output_im[class_index] = tmp / 169.0;
    }
}
 
