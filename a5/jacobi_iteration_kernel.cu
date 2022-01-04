#include "jacobi_iteration.h"

__device__ void lock(int *mutex) 
{	  
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}
__device__ void unlock(int *mutex) 
{
    atomicExch(mutex, 0);
    return;
}

__global__ void jacobi_iteration_kernel_naive(float *d_A, float * d_B, float *curr_X, float *prev_X, int *input_elements, double *ssd)
{
   
    int tid = threadIdx.y;
    double sum = 0;
    int num_elements = *input_elements;

    int i = (blockIdx.y * THREAD_BLOCK_SIZE) + tid; 
    for(int j = 0; j < num_elements; j++){
        if (i != j)
            sum += d_A[i * num_elements + j] * curr_X[j];
    }

    prev_X[i] = (d_B[i] - sum)/d_A[i * num_elements + i];

    ssd[i] = (curr_X[i] - prev_X[i]) * (curr_X[i] - prev_X[i]);

    return;
}

__global__ void jacobi_iteration_kernel_optimized(float *d_A, float *opt, float *d_B, float *out_X, double *ssd, int *mutex)
{
    __shared__ double ssd_device[THREAD_BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int blkid = blockIdx.x * blockDim.x + threadIdx.x;

    //reset ssd = 0
    if (blkid == 0)
        *ssd = 0.0;

    // begin iteration
    double sum = -d_A[blkid * MATRIX_SIZE + blkid] * opt[blkid];
    for (int j = 0; j < MATRIX_SIZE; j++) {
        sum += d_A[blkid + MATRIX_SIZE * j] * opt[j];
    }
    out_X[blkid] = (d_B[blkid] - sum)/d_A[blkid * MATRIX_SIZE + blkid];

    if (blkid < MATRIX_SIZE)
        ssd_device[tid] = (out_X[blkid] - opt[blkid]) * (out_X[blkid] - opt[blkid]);
    else
        ssd_device[tid] = 0.0;

    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
       if(tid < stride)
          ssd_device[tid] += ssd_device[tid + stride];
       __syncthreads();
    }

    if (tid == 0) {
       lock(mutex);
       *ssd += ssd_device[0];
       unlock(mutex);
    }

    return;
}  

