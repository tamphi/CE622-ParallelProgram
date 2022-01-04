/* Reference code implementing the box blur filter.

    Build and execute as follows: 
        make clean && make 
        ./blur_filter size

    Author: Naga Kandasamy
    Date created: May 3, 2019
    Date modified: February 15, 2021

    Student name(s): FIXME
    Date modified: FIXME
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* #define DEBUG */

//#define CONCISE
/* Include the kernel code */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t);
int check_results(const float *, const float *, int, float);
void print_image(const image_t);
image_t allocate_image_on_device (image_t);
void copy_image_to_device(image_t, image_t);
void copy_image_from_device(image_t, image_t);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s size\n", argv[0]);
        fprintf(stderr, "size: Height of the image. The program assumes size x size image.\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images */
    int size = atoi(argv[1]);

    fprintf(stderr, "Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *)malloc(sizeof(float) * size * size);
    out_gold.element = (float *)malloc(sizeof(float) * size * size);
    out_gpu.element = (float *)malloc(sizeof(float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand(time(NULL));
    int i;
    for (i = 0; i < size * size; i++)
        in.element[i] = rand()/(float)RAND_MAX -  0.5;

    /* Calculate the blur on the CPU. The result is stored in out_gold. */
    fprintf(stderr, "Calculating blur on the CPU\n");

    struct timeval start, stop;

    gettimeofday(&start,NULL);
    /* Compute Jacobi solution using reference code */
    compute_gold(in, out_gold); 
    gettimeofday(&stop,NULL);

    float gold_time = (float)(stop.tv_sec - start.tv_sec\
                              + (stop.tv_usec - start.tv_usec)/(float)1000000);

    #ifdef DEBUG 
    print_image(in);
    print_image(out_gold);
    #endif

    /* FIXME: Calculate the blur on the GPU. The result is stored in out_gpu. */
    fprintf(stderr, "Calculating blur on the GPU\n");
    compute_on_device(in, out_gpu);

    /* Check CPU and GPU results for correctness */
    fprintf(stderr, "Checking CPU and GPU results\n");
    int num_elements = out_gold.size * out_gold.size;
    float eps = 1e-6;    /* Do not change */
    int check;
    check = check_results(out_gold.element, out_gpu.element, num_elements, eps);
    if (check == 0) 
       fprintf(stderr, "TEST PASSED\n");
    else
       fprintf(stderr, "TEST FAILED\n");
   
    #ifdef DEBUG
    print_image(out_gpu);
    #endif       

   // FILE *fp = fopen("result.txt","a");
   fprintf(stdout,"gold time: %f\n", gold_time);
   // fclose(fp);
   /* Free data structures on the host */
   free((void *)in.element);
   free((void *)out_gold.element);
   free((void *)out_gpu.element);

    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU */
void compute_on_device(const image_t in, image_t out)
{
   /* Allocate memory on device */
    image_t d_IN = allocate_image_on_device(in);  
    image_t d_OUT = allocate_image_on_device(out);

    /* Copy image to device */
    copy_image_to_device(d_IN, in); 	         
    copy_image_to_device(d_OUT, out);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid(d_IN.size/threads.x, d_OUT.size/threads.y);

    struct timeval start, stop;
    gettimeofday(&start,NULL);
    /*Launch kernel */
    blur_filter_kernel<<<grid, threads>>>(d_IN.element, d_OUT.element, d_IN.size);
    gettimeofday(&stop,NULL);
    
    float cuda_time = (float)(stop.tv_sec - start.tv_sec\
                              + (stop.tv_usec - start.tv_usec)/(float)1000000);
    fprintf(stdout,"cuda time: %f\n", cuda_time);
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    copy_image_from_device(out,d_OUT);
    cudaFree(d_IN.element);
    cudaFree(d_OUT.element);
    return;
}

/* Allocate matrix on device */
image_t allocate_image_on_device (image_t M) {
	image_t Mdevice = M;
	int size = M.size * M.size * sizeof(float);
	cudaMalloc((void**)&Mdevice.element, size);
	return Mdevice;
}

/*Copy from host to device */
void copy_image_to_device(image_t Mdevice, image_t Mhost) {
   int size = Mhost.size * Mhost.size * sizeof(float);
   Mdevice.size = Mhost.size;
   cudaMemcpy(Mdevice.element, Mhost.element, size, cudaMemcpyHostToDevice);
}

/*Copy from device to host */
void copy_image_from_device(image_t Mhost, image_t Mdevice) {
   int size = Mdevice.size * Mdevice.size * sizeof(float);
   cudaMemcpy(Mhost.element, Mdevice.element, size, cudaMemcpyDeviceToHost);
}


/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements, float eps) 
{
    int i;
    for (i = 0; i < num_elements; i++)
        if (fabsf((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return 1;
    
    return 0;
}

/* Print out the imagee contents */
void print_image(const image_t img)
{
    int i, j;
    float val;
    for (i = 0; i < img.size; i++) {
        for (j = 0; j < img.size; j++) {
            val = img.element[i * img.size + j];
            printf("%0.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
}
