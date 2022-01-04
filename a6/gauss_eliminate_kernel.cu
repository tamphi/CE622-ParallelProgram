 /* Device code. */
#include "gauss_eliminate.h"

__global__ void division__kernel(float *device_U, int row, int size)
{
	
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int check = blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float pivot = device_U[row*size + row];
    while((tid + (row+1)) < size){
        device_U[row*size + (row+1) + tid] = __fdiv_rn( device_U[row*size + (row+1) + tid] , pivot );
        tid = tid + stride;
    }

    //if(blockIdx.x == 0 && threadIdx.x == 0){
    if(!check){
        device_U[row*size + row] = 1;
    }
    __syncthreads();
}

__global__ void elimination_kernel(float *device_U, int row, int size)
{

    int curr = (row+1) + blockIdx.x;
    while (curr<size){    
        int tid = threadIdx.x;
        while((tid + (row+1)) < size){  
           int offset = curr*size + row +1 +tid;
           int offset2 = row*size + row +1 +tid;
           int offset3 = curr*size + row;
           device_U[offset] = __fsub_rn( device_U[offset] , __fmul_rn( device_U[offset3] , device_U[offset2] ) );
            tid = tid + blockDim.x;
        }
        __syncthreads();
        if(threadIdx.x == 0){
            device_U[curr*size + row] = 0;
        }
        curr = curr + gridDim.x;
    }
    __syncthreads();
}
