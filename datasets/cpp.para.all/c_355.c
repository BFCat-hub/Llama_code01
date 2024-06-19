#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void CrossEntropyLoss_forward(float *logits_data, float *logits_grad, float *loss, int *truth, int training, int num_classes, int size, int grad_size) {
    float total_loss = 0;
    int count = 0;

    for (int i = 0; i < size / num_classes; i++) {
        if (truth[i] < 0) continue;

        count++;
        float *logit = &logits_data[i * num_classes];
        float max_logit = -1e30, sum_exp = 0;

        for (int j = 0; j < num_classes; j++)
            max_logit = fmax(max_logit, logit[j]);

        for (int j = 0; j < num_classes; j++) {
            logit[j] -= max_logit;
            sum_exp += expf(logit[j]);
        }

        total_loss += logf(sum_exp) - logit[truth[i]];

        if (training) {
            for (int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits_grad[i * num_classes + j] = prob;
            }

            logits_grad[i * num_classes + truth[i]] -= 1.0;
        }
    }

    *loss = total_loss / count;

    if (training) {
        for (int i = 0; i < grad_size; i++)
            logits_grad[i] /= count;
    }
}

int main() {
    // Test CrossEntropyLoss_forward function with a simple example
    int num_classes = 3;
    int size = 9;  // num_classes * num_samples
    int grad_size = num_classes * (size / num_classes);
    int *truth = (int *)malloc(size / num_classes * sizeof(int));

    float *logits_data = (float *)malloc(size * sizeof(float));
    float *logits_grad = (float *)malloc(grad_size * sizeof(float));
    float loss;

    // Initialize truth array and logits_data (you may modify this part based on your actual data)
    for (int i = 0; i < size / num_classes; i++) {
        truth[i] = i % num_classes;
    }

    for (int i = 0; i < size; i++) {
        logits_data[i] = i * 0.1;  // Example: Replace with your actual data
    }

    // Call the CrossEntropyLoss_forward function
    CrossEntropyLoss_forward(logits_data, logits_grad, &loss, truth, 1, num_classes, size, grad_size);

    // Display the results (you may modify this part based on your actual data)
    printf("Loss: %f\n", loss);

    printf("Gradient:\n");
    for (int i = 0; i < grad_size; i++) {
        printf("%f ", logits_grad[i]);
    }

    free(truth);
    free(logits_data);
    free(logits_grad);

    return 0;
}
 
