#ifndef _JACOBI_SOLVER_H_
#define _JACOBI_SOLVER_H_

#define THRESHOLD 1e-5      /* Threshold for convergence */
#define MIN_NUMBER 2        /* Min number in the A and b matrices */
#define MAX_NUMBER 10       /* Max number in the A and b matrices */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <semaphore.h>
#include <pthread.h>
#include <semaphore.h>
/* Matrix structure declaration */
typedef struct matrix_s {
    unsigned int num_columns;   /* Matrix width */
    unsigned int num_rows;      /* Matrix height */ 
    float *elements;
}  matrix_t;

/* Structure that defines the barrier */
typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;


/* Data structure defining what to pass to each worker thread */
typedef struct thread_data_s {
    int tid;                        /* The thread ID */
    int num_threads;                /* Number of threads in the worker pool */
    int offset;
    int chunk_size;
    matrix_t * a;                /* Pointer to vector y */
    matrix_t * x;                /* Pointer to vector x */ 
    matrix_t * b;            /* Starting address of the partial_sum array */
    matrix_t * prevX;            /* Starting address of the partial_sum array */
    pthread_mutex_t *mutex_for_sum;
    float *ssd;
    int *done; 
} thread_data_t;

/* Function prototypes */
matrix_t allocate_matrix(int, int, int);
extern void compute_gold(const matrix_t, matrix_t, const matrix_t, int);
extern void display_jacobi_solution(const matrix_t, const matrix_t, const matrix_t);
int check_if_diagonal_dominant(const matrix_t);
matrix_t create_diagonally_dominant_matrix(int, int);
float * compute_using_pthreads_v1(const matrix_t, matrix_t, const matrix_t, int Max_iter, int num_threads);
void compute_using_pthreads_v2(const matrix_t, matrix_t, const matrix_t, int Max_iter, int num_threads);
void print_matrix(const matrix_t);
float get_random_number(int, int);
void barrier_sync(barrier_t *, int, int);
void* jacobi_v1(void * args);
void* jacobi_v2(void * args);


#endif /* _JACOBI_SOLVER_H_ */

