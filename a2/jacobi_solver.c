/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: January 28, 2021
 *
 * Student name(s): FIXME
 * Date modified: FIXME
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -std=c99 -Wall -lpthread -lm
*/

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "jacobi_solver.h"
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>
#include <sys/time.h>


/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */
barrier_t barrier1;  
barrier_t barrier2;  


int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        fprintf(stderr, "num-threads: number of worker threads to create\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x_v1;      /* Solution computed by pthread code using chunking */
    matrix_t mt_solution_x_v2;      /* Solution computed by pthread code using striding */

	/* Generate diagonally dominant matrix */
    //fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        //fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x_v1 = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x_v2 = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(reference_x);
   
#else
    #define max_iter  10000
#endif

    struct timeval start, stop;	

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    
    
     /* Maximum number of iterations to run */
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("\n");

#ifdef DEBUG
	print_matrix(reference_x);
#endif

	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x_v1 and mt_solution_x_v2.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using chunking\n");
	gettimeofday(&start, NULL);
    compute_using_pthreads_v1(A, mt_solution_x_v1, B, max_iter, num_threads);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x_v1, B); /* Display statistics */
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));
	printf("\n");
    
#ifdef DEBUG
	print_matrix(mt_solution_x_v1);
#endif

    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using striding\n");
	gettimeofday(&start, NULL);
    compute_using_pthreads_v2(A, mt_solution_x_v2, B, max_iter, num_threads);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x_v2, B); /* Display statistics */
    printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x_v1.elements);
    free(mt_solution_x_v2.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using chunking. 
 * Result must be placed in mt_sol_x_v1. */
float * compute_using_pthreads_v1(const matrix_t A, matrix_t mt_sol_x_v1, const matrix_t B, int Max_iter, int num_threads)
{
    pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t));
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    int i;
    int done = 0;
    int chunk_sz = (int)floor((float)B.num_rows/(float)num_threads); //Num row per thread
    float ssd = 0;
    
    barrier1.counter = 0;
    sem_init(&barrier1.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&barrier1.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

    barrier2.counter = 0;
    sem_init(&barrier2.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&barrier2.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

    pthread_mutex_t mutex_for_sum;                                                  /* Lock for the shared variable sum */
    pthread_mutex_init(&mutex_for_sum, NULL);  

    matrix_t * pA = &A;
    matrix_t * pB = &B;
    matrix_t * prevX = &mt_sol_x_v1;
    matrix_t X = allocate_matrix(B.num_rows, 1, 0);
    matrix_t * pX = &X; 


    thread_data_t *tdata = (thread_data_t *)malloc(sizeof(thread_data_t)*num_threads);
    for (i = 0; i < num_threads; i++) {
        tdata[i].tid = i; 
        tdata[i].num_threads = num_threads;
        tdata[i].a = pA; 
        tdata[i].x = pX;
        tdata[i].prevX = prevX; 
        tdata[i].b = pB; 
        tdata[i].chunk_size = chunk_sz;
        tdata[i].done = &done;
        tdata[i].mutex_for_sum = &mutex_for_sum;
        tdata[i].ssd = &ssd;
    }

    for (i = 0; i < num_threads; i++){
        pthread_create(&thread_id[i], &attr, jacobi_v1, (void *)&tdata[i]);
    }

    // void * buffer = allocate_matrix;				 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

    /* Free stuffs */
    free((void *)thread_id);
    free((void *)tdata);
    return (mt_sol_x_v1.elements != X.elements)? X.elements : prevX->elements;

}

void* jacobi_v1(void * args) {
    
    thread_data_t *tdata = (thread_data_t *)args;
    tdata->offset = tdata->tid * tdata->chunk_size;
    int n = tdata->b->num_rows;
    int i, j; 
    int num_iter = 0;
    matrix_t temp;
    int end_row = (tdata->tid < (tdata->num_threads - 1)) ? (tdata->offset + tdata->chunk_size) : n;
    float mse = 0;

    while (*(tdata->done) == 0) {
        float local_ssd = 0;
        for (i = tdata->offset; i < end_row; i++) {
            // tdata->y[i] = tdata->a* tdata->x[i] + tdata->y[i];
            double sum = -tdata->a->elements[i * n + i] * tdata->prevX->elements[i];
            for (j = 0; j < n; j++) {
                sum += tdata->a->elements[i * n + j] * tdata->prevX->elements[j];
            }
            tdata->x->elements[i] = (tdata->b->elements[i] - sum)/tdata->a->elements[i*n + i];
            local_ssd += (tdata->x->elements[i] - tdata->prevX->elements[i])*(tdata->x->elements[i] - tdata->prevX->elements[i]);
        }
        assert(tdata->mutex_for_sum != NULL);
        pthread_mutex_lock(tdata->mutex_for_sum);
        *(tdata->ssd) += local_ssd;
        pthread_mutex_unlock(tdata->mutex_for_sum);

        barrier_sync(&barrier1, tdata->tid, tdata->num_threads);   
        if (tdata->tid == 0) {
            temp = *(tdata->prevX);
            *(tdata->prevX) = *(tdata->x);
            *(tdata->x) = temp;        
            num_iter++;
            mse = sqrt(*(tdata->ssd));
            *(tdata->ssd) = 0;
            if ((mse <= THRESHOLD) || (num_iter == max_iter)) {
                *(tdata->done) = 1;
            }
        }
        barrier_sync(&barrier2, tdata->tid, tdata->num_threads);   
    }
    pthread_exit(NULL);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using striding. 
 * Result must be placed in mt_sol_x_v2. */
void compute_using_pthreads_v2(const matrix_t A, matrix_t mt_sol_x_v2, const matrix_t B, int Max_iter, int num_threads)
{
    pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t));
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    int i;
    int done = 0;
    // int chunk_sz = (int)floor((float)B.num_rows/(float)num_threads); //Num row per thread
    float ssd = 0;
    
    barrier1.counter = 0;
    sem_init(&barrier1.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&barrier1.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

    barrier2.counter = 0;
    sem_init(&barrier2.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&barrier2.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

    pthread_mutex_t mutex_for_sum;                                                  /* Lock for the shared variable sum */
    pthread_mutex_init(&mutex_for_sum, NULL);  

    matrix_t * pA = &A;
    matrix_t * pB = &B;
    matrix_t * prevX = &mt_sol_x_v2;
    matrix_t X = allocate_matrix(B.num_rows, 1, 0);
    matrix_t * pX = &X; 


    thread_data_t *tdata = (thread_data_t *)malloc(sizeof(thread_data_t)*num_threads);
    for (i = 0; i < num_threads; i++) {
        tdata[i].tid = i; 
        tdata[i].num_threads = num_threads;
        tdata[i].a = pA; 
        tdata[i].x = pX;
        tdata[i].prevX = prevX; 
        tdata[i].b = pB; 
        tdata[i].done = &done;
        tdata[i].mutex_for_sum = &mutex_for_sum;
        tdata[i].ssd = &ssd;
    }

    for (i = 0; i < num_threads; i++){
        pthread_create(&thread_id[i], &attr, jacobi_v2, (void *)&tdata[i]);
    }

    // void * buffer = allocate_matrix;				 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

    /* Free stuffs */
    free((void *)thread_id);
    free((void *)tdata);
    // return (mt_sol_x_v2.elements != X.elements)? X.elements : prevX->elements;
}

void* jacobi_v2(void * args) {
    
    thread_data_t *tdata = (thread_data_t *)args;
    int n = tdata->b->num_rows;
    int i, j; 
    int num_iter = 0;
    matrix_t temp;
    float mse = 0;

    while (*(tdata->done) == 0) {
        float local_ssd = 0;
        i = tdata->tid;
        while (i < n) {
            double sum = -tdata->a->elements[i * n + i] * tdata->prevX->elements[i];
            for (j = 0; j < n; j++) {
                sum += tdata->a->elements[i * n + j] * tdata->prevX->elements[j];
            }
            tdata->x->elements[i] = (tdata->b->elements[i] - sum)/tdata->a->elements[i*n + i];
            local_ssd += (tdata->x->elements[i] - tdata->prevX->elements[i])*(tdata->x->elements[i] - tdata->prevX->elements[i]);
            i += tdata->num_threads; 
        }

        assert(tdata->mutex_for_sum != NULL);
        pthread_mutex_lock(tdata->mutex_for_sum);
        *(tdata->ssd) += local_ssd;
        pthread_mutex_unlock(tdata->mutex_for_sum);

        barrier_sync(&barrier1, tdata->tid, tdata->num_threads);   
        if (tdata->tid == 0) {
            temp = *(tdata->prevX);
            *(tdata->prevX) = *(tdata->x);
            *(tdata->x) = temp;        
            num_iter++;
            mse = sqrt(*(tdata->ssd));
            // fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse); 
            *(tdata->ssd) = 0;
            if ((mse <= THRESHOLD) || (num_iter == max_iter)) {
                *(tdata->done) = 1;
            }
        }
        barrier_sync(&barrier2, tdata->tid, tdata->num_threads);   
    }
    pthread_exit(NULL);
}





/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



void barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
    int i;

    sem_wait(&(barrier->counter_sem));
    /* Check if all threads before us, that is num_threads - 1 threads have reached this point. */	  
    if (barrier->counter == (num_threads - 1)) {
        barrier->counter = 0; /* Reset counter value */
        sem_post(&(barrier->counter_sem)); 	 
        /* Signal blocked threads that it is now safe to cross the barrier */
        //printf("Thread number %d is signalling other threads to proceed\n", tid); 
        for (i = 0; i < (num_threads - 1); i++)
            sem_post(&(barrier->barrier_sem));
    } 
    else { /* There are threads behind us */
        barrier->counter++;
        sem_post(&(barrier->counter_sem));
        sem_wait(&(barrier->barrier_sem)); /* Block on the barrier semaphore */
    }

    return;
}
