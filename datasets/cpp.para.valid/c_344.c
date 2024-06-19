#include <stdio.h>

long int maxValExtractArray(float *normM_aux, long int *b_pos, long int b_pos_size) {
    float max_val = -1;
    long int pos = -1;

    for (long int i = 0; i < b_pos_size; i++) {
        if (normM_aux[b_pos[i]] > max_val) {
            max_val = normM_aux[b_pos[i]];
            pos = i;
        }
    }

    return pos;
}

int main() {
    // Test maxValExtractArray function with a simple example
    long int b_pos_size = 5;
    float normM_aux[] = {1.2, 3.5, 2.8, 5.1, 4.2};
    long int b_pos[] = {2, 0, 4, 1, 3};

    long int result = maxValExtractArray(normM_aux, b_pos, b_pos_size);

    // Display the result
    printf("Position of maximum value: %ld\n", result);

    return 0;
}
 
