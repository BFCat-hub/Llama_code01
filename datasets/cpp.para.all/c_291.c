#include <stdio.h>

// Function prototype
void Gather_cpu(const int *input, float *output, int input_size, const float *data, int count, int dim, int data_offset);

int main() {
    // Example data
    int input_size = 3;
    int input[] = {2, 0, 1}; // Assuming input_size * dim = 9
    float output[input_size * dim];
    int count = 2;
    int dim = 3;
    int data_offset = 1;
    float data[] = {1.1, 1.2, 1.3,
                    2.1, 2.2, 2.3,
                    3.1, 3.2, 3.3,
                    4.1, 4.2, 4.3};

    // Call the function
    Gather_cpu(input, output, input_size, data, count, dim, data_offset);

    // Display the results
    printf("Gathered Data:\n");
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%.2f ", output[i * dim + j]);
        }
        printf("\n");
    }

    return 0;
}

// Function definition
void Gather_cpu(const int *input, float *output, int input_size, const float *data, int count, int dim, int data_offset) {
    int index;
    for (index = 0; index < input_size * dim; index++) {
        const int input_id = input[index / dim];
        const int pos = index % dim;
        if (input_id < count + data_offset && input_id >= data_offset) {
            output[index] = data[input_id * dim + pos];
        }
    }
}
 
