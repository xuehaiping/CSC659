#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#define MAXSIZE 16384
/* LU: Compute LU factorization of A
 * Upper triangular part of A is overwritten with U
 * Strict lower triangular part of A is overwritten with L
 * L has ones on diagonal, but these are not stored
 */
 //NOTE:good!
void LU(int n, double A[][n])
{

  int i, j, k;
  double pivot;

  #pragma omp parallel default(none) private(i, j, k, pivot)  shared(A, n)
  {
      for (k = 0; k < n-1; k++) {
          pivot = A[k][k];
          #pragma omp for
          for (i = k+1; i < n; i++) {
              A[i][k] = A[i][k]/pivot;
          }
          #pragma omp for
          for (i = k+1; i < n; i++)
              for (j = k+1; j < n; j++)
                  A[i][j] = A[i][j] - A[i][k]*A[k][j];
      }
  }
}
/* solve_U: Solve strict lower triangular system Lx = b
 * Routine is called with x=b
 * L has ones on diagonal (not stored);
 * Upper triangle of L is ignored
 */
 //NOTE:good!
void solve_L(int n, double x[], double L[][n])
{
    int i, j;
    #pragma omp parallel default(none) private(i, j) shared(x, n, L)
    {
      for (i = 1; i < n; i++) {
        #pragma omp for
          for (j = i; j < n; j++)
              x[j] = x[j] - L[j][i-1]*x[i-1];
      }
    }
}

/* solve_U: Solve upper triangular system Ux = b
 * Routine is called with x=b
 * Lower triangle of U is ignored
 */
 //NOTE:good!
void solve_U(int n, double x[], double U[][n])
{
    int i, j;

    for (i = n-1; i > 0; i--) {
        x[i] = x[i]/U[i][i];
        #pragma omp parallel for default(none) private(j) firstprivate(i) shared(n, x, U)
        for (j = 0; j < i; j++)
            x[j] = x[j] - U[j][i]*x[i];
    }
    x[0] = x[0]/U[0][0];

}
/* matvec: Compute y = A*x
 */
  //NOTE:good!
void matvec(int n, double y[], double x[], double A[][n])
{
    int i, j;

    #pragma omp parallel for default(none) private(i, j) shared(n, x, y, A)
    for (i = 0; i < n; i++) {
        y[i] = 0.0;
        for (j = 0; j < n; j++)
            y[i] = y[i] + A[i][j]*x[j];
    }
}
/* saxpy: Compute y = x + alpha * y
 */
 //NOTE:good!
void saxpy(int n, double alpha, double y[], double x[])
{
    int i;
    #pragma omp parallel for default(none) private(i) shared(n, alpha, y, x)
    for (i = 0; i < n; i++) y[i] = x[i] + alpha*y[i];
}
/* nrm: Compute 2-norm of x
 */
 //NOTE:good!
double norm(int n, double x[])
{
    int i;
    double sum = 0.0;
    double nrm;

    #pragma omp parallel for default(none) private(i) shared(n, x) reduction(+:sum)
    for (i = 0; i < n; i++) sum = sum + x[i]*x[i];

    return sqrt(sum);
}

void initialize_vector(int n, double x[])
{
    int i;
    for (i = 0; i < n; i++) x[i] = 1.0;
}

void initialize_matrix(int n, double A[][n])
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i][j] = 1.0/(1.0+abs(i-j));
}
main(int argc, char **argv)
{
    double nrm;
    enum timer {tinit, tLU, tsolve_L, tsolve_U, tmatvec, tsaxpy, tnorm};
    double start, timing[tnorm+1];
    double A[MAXSIZE][MAXSIZE], A0[MAXSIZE][MAXSIZE];
    double x[MAXSIZE], b[MAXSIZE], y[MAXSIZE];
    int n;
    /* Check input parameters */
    n = atoi(argv[1]);
    if (n > MAXSIZE) {
        printf("n must be less than %d ... aborting\n", MAXSIZE);
        exit(0);
    }
    if (n <= 0) {
        printf("n must be a positive integer ... aborting\n");
        exit(0);
    }

    start = omp_get_wtime();    /* Start timing */
    initialize_matrix(n, A);    /* Initialize matrix */
    initialize_matrix(n, A0);   /* Copy A in A0 */

    initialize_vector(n, b);    /* Initialize vector */
    initialize_vector(n, x);    /* Copy b in x */
    timing[tinit] = omp_get_wtime() - start;

    start = omp_get_wtime();    /* Start timing */
    LU(n, A);                   /* LU factorization */
    timing[tLU] = omp_get_wtime() - start;

    start = omp_get_wtime();    /* Start timing */
    solve_L(n, x, A);           /* Compute x = inv(L)x */
    timing[tsolve_L] = omp_get_wtime() - start;

    start = omp_get_wtime();    /* Start timing */
    solve_U(n, x, A);           /* Compute x = inv(L)x */
    timing[tsolve_U] = omp_get_wtime() - start;

    start = omp_get_wtime();    /* Start timing */
    matvec(n, y, x, A0);        /* Check solution: y = A*x */
    timing[tmatvec] = omp_get_wtime() - start;

    start = omp_get_wtime();    /* Start timing */
    saxpy(n, -1.0, y, b);       /* Check solution: y = b - y */
    timing[tsaxpy] = omp_get_wtime() - start;

    start = omp_get_wtime();    /* Start timing */
    nrm = norm(n, y);           /* Check solution: nrm = ||y|| */
    timing[tnorm] = omp_get_wtime() - start;

    printf("Residual norm: ||b-Ax|| = %e\n", nrm);
    printf("Size - n:       %10d\n", n);
    printf("Time - init:    %10.4e\n", timing[tinit]);
    printf("Time - LU:      %10.4e\n", timing[tLU]);
    printf("Time - solve_L: %10.4e\n", timing[tsolve_L]);
    printf("Time - solve_U: %10.4e\n", timing[tsolve_U]);
    printf("Time - matvec:  %10.4e\n", timing[tmatvec]);
    printf("Time - saxpy:   %10.4e\n", timing[tsaxpy]);
    printf("Time - norm:    %10.4e\n", timing[tnorm]);
/*
    printf("%10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %12.2e\n",
            timing[tLU], timing[tsolve_L], timing[tsolve_U],
            timing[tmatvec], timing[tsaxpy], timing[tnorm],
            nrm);
*/
}
