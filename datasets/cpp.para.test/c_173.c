#include <stdio.h>

void yuv2rgb_kernel(int img_size, unsigned char *gpu_img_in_y, unsigned char *gpu_img_in_u, unsigned char *gpu_img_in_v, 
                    unsigned char *gpu_img_out_r, unsigned char *gpu_img_out_g, unsigned char *gpu_img_out_b);

int main() {
    // Example parameters
    const int img_size = 10;
    unsigned char gpu_img_in_y[img_size];
    unsigned char gpu_img_in_u[img_size];
    unsigned char gpu_img_in_v[img_size];
    unsigned char gpu_img_out_r[img_size];
    unsigned char gpu_img_out_g[img_size];
    unsigned char gpu_img_out_b[img_size];

    // Initialize your input YUV data here (replace with your actual data)
    for (int i = 0; i < img_size; ++i) {
        gpu_img_in_y[i] = 100;
        gpu_img_in_u[i] = 50;
        gpu_img_in_v[i] = 150;
    }

    // Call yuv2rgb_kernel function
    yuv2rgb_kernel(img_size, gpu_img_in_y, gpu_img_in_u, gpu_img_in_v, gpu_img_out_r, gpu_img_out_g, gpu_img_out_b);

    // Print the results or add further processing as needed
    for (int i = 0; i < img_size; ++i) {
        printf("(%d, %d, %d) -> (%d, %d, %d)\n", gpu_img_in_y[i], gpu_img_in_u[i], gpu_img_in_v[i],
               gpu_img_out_r[i], gpu_img_out_g[i], gpu_img_out_b[i]);
    }

    return 0;
}

void yuv2rgb_kernel(int img_size, unsigned char *gpu_img_in_y, unsigned char *gpu_img_in_u, unsigned char *gpu_img_in_v, 
                    unsigned char *gpu_img_out_r, unsigned char *gpu_img_out_g, unsigned char *gpu_img_out_b) {
    int rt, gt, bt;
    int rt2, gt2, bt2;
    
    for (int index = 0; index < img_size; index++) {
        rt = (int)(gpu_img_in_y[index] + 1.402 * (gpu_img_in_v[index] - 128));
        gt = (int)(gpu_img_in_y[index] - 0.344 * (gpu_img_in_u[index] - 128) - 0.714 * (gpu_img_in_v[index] - 128));
        bt = (int)gpu_img_in_y[index] + 1.772 * (gpu_img_in_u[index] - 128);

        rt2 = (rt > 255) ? 255 : rt;
        gt2 = (gt > 255) ? 255 : gt;
        bt2 = (bt > 255) ? 255 : bt;

        gpu_img_out_r[index] = (rt2 < 0) ? 0 : rt2;
        gpu_img_out_g[index] = (gt2 < 0) ? 0 : gt2;
        gpu_img_out_b[index] = (bt2 < 0) ? 0 : bt2;
    }
}
