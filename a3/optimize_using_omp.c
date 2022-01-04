/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: February 5, 2021
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "pso.h"
#include <sys/time.h>
#include <omp.h>

/* Solve PSO */
int pso_solve_omp(char *function, swarm_t *swarm, 
                    float xmax, float xmin, int max_iter, int num_threads)
{   

    int i, j, iter, g;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;

    /* thread safe rand_r() */
    unsigned int seedp;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    g = -1;

    
    while (iter < max_iter) {
       /* begin multi thread */
#pragma omp parallel num_threads(num_threads) shared(w,c1,c2,xmax,xmin,function) private(i,j,r1,r2,gbest,particle,curr_fitness,seedp)
       {
       seedp = time(NULL) + omp_get_thread_num();
#pragma omp for
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand_r(&seedp)/(float)RAND_MAX;
                r2 = (float)rand_r(&seedp)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = w * particle->v[j]\
                                 + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                    particle->v[j] = uniform_omp(-fabsf(xmax - xmin), fabsf(xmax - xmin), seedp);

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax)
                    particle->x[j] = xmax;
                if (particle->x[j] < xmin)
                    particle->x[j] = xmin;
            } /* State update */
            
            /* Evaluate current fitness */
            pso_eval_fitness(function, particle, &curr_fitness);

            /* Update pbest */
            if (curr_fitness < particle->fitness) {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        } /* Particle loop */

       } /* end of pragma */

        /* Identify best performing particle */
        g = pso_get_best_fitness_omp(swarm,num_threads);
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            particle->g = g;
        }

#ifdef SIMPLE_DEBUG
        /* Print best performing particle */
        fprintf(stderr, "\nIteration %d:\n", iter);
        pso_print_particle(&swarm->particle[g]);
#endif
        iter++;
    } /* End of iteration */
    return g;

} /* End of pso_solv_omp */


int optimize_using_omp(char *function, int dim, int swarm_size, 
                       float xmin, float xmax, int num_iter, int num_threads)
{  
     /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));
    swarm = pso_init_omp(function, dim, swarm_size, xmin, xmax,num_threads);
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

#ifdef VERBOSE_DEBUG
    pso_print_swarm(swarm);
#endif

    /* Solve PSO */
    int g; 
    g = pso_solve_omp(function, swarm, xmax, xmin, num_iter, num_threads);
    if (g >= 0) {
//        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}
