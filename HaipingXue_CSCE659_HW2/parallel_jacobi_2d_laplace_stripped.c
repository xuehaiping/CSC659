#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

// =============================================================================
// Compilation and Interactive Execution
//     module load intel/compilers
//     module load openmpi
//     mpicc -o parallel_jacobi_2d_laplace.exe parallel_jacobi_2d_laplace.c
//     mpirun -np 4 parallel_jacobi_2d_laplace.exe 64 64 2 2
//
// Mesh
//   Domain is discretized by a uniform N x M mesh or grid
//   Local mesh on a process has size n x m
//   Local mesh is stored in U[0..n+1][0..m+1] where a single element layer
//   along the boundary stores boundary values
//   - U[0][*] stores the left boundary
//   - U[n+1][*] stores the right boundary
//   - U[*][0] stores the below boundary
//   - U[*][m+1] store the above boundary
//   Boundary values initialized by values from adjacent neighbor process;
//   or by domain boundary values if neighbor is absent
//
// Process data
//   - Processes are organized in a 2D grid of size px x py processes
//   - Process with coordinates (my_row,my_col) has the rank (i.e., id)
//     my_id = my_col + my_row * px
//   - Process has four neighbors whose ids are stored in my_left, my_right,
//     my_above, and my_below (a non-existent neighbor is given id NO_ONE)
//
// MPI data
//   buf is a buffer used by MPI_Send and MPI_Receive
//
// Miscellaneous
//   verbose: set to a value greater than 0 for useful messages
//
// =============================================================================
double ** U;
double ** Utemp;
int N;
int M;
int n;
int m;
int px;
int py;

int numprocs;
int my_id;
int my_row;
int my_col;
int my_left;
int my_right;
int my_above;
int my_below;
double * buf;
#define NO_ONE -1
#define TAG 1

int verbose;

#define TOL 1.0e-4

// -----------------------------------------------------------------------------
// Process routines
// -----------------------------------------------------------------------------
// Determine neighbor processes of a process
void initialize_process_info() {
    my_row = my_id / px;
    my_col = my_id % px;
    if (my_col == 0) {			// Left neighbor
	my_left = NO_ONE;
    } else {
	my_left = my_id - 1;
    }
    if (my_col == px-1) {		// Right neighbor
	my_right = NO_ONE;
    } else {
	my_right = my_id + 1;
    }
    if (my_row == py-1) {		// Above neighbor
	my_above = NO_ONE;
    } else {
	my_above = my_id + px;
    }
    if (my_row == 0) {			// Below neighbor
	my_below = NO_ONE;
    } else {
	my_below = my_id - px;
    }
}

// -----------------------------------------------------------------------------
// MPI support routines
// -----------------------------------------------------------------------------
// Allocate buffer of size max(m+2,n+2) doubles for communicating boundary
// values among processors
double * allocate_mpi_buffer (int n, int m) {
    double * buffer;
    int bufsize = n+2;
    if (m > n) bufsize = m+2;
    buffer = (double *) calloc(bufsize, sizeof(double));
    if (buffer == NULL) {
	printf("allocate_mpi_buffer: cannot allocate memory for mpi buffer ... exiting\n");
	exit(1);
    }
    return buffer;
}
// -----------------------------------------------------------------------------
// Free buffer
void free_mpi_buffer(double * buffer) {
    free(buffer);
}
// -----------------------------------------------------------------------------
// Mesh routines
// -----------------------------------------------------------------------------
// Allocate an n x m 2D array of doubles
double ** allocate_2d_array (int n, int m) {
    double ** A;
    int i, j;
    int error = 0;
    A = calloc(n,sizeof(double *));
    if (A == NULL) {
	error = 1;
    } else {
	for (i = 0; i < n; i++) {
	    A[i] = (double *) calloc(m,sizeof(double));
	    if (A[i] == NULL) {
		error = 1;
		break;
	    }
	}
    }
    if (error) {
	printf("allocate_2d_array: cannot allocate memory for A ... exiting\n");
	exit(1);
    }
    return A;
}
// -----------------------------------------------------------------------------
// Free an n x m 2D array of doubles
void free_2d_array (int n, int m, double ** A) {
    int i;
    for (i = 0; i < n; i++) {
	free(A[i]);
    }
    free(A);
}
// -----------------------------------------------------------------------------
// Initialize 2D mesh to 0.0
void initialize_mesh (int n, int m, double *A[]) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            A[i][j] = 0.0;
}
// -----------------------------------------------------------------------------
// Output solution to OUTPUT
void output_solution() {
    int i, j, ip, jp, i_offset, j_offset;
    FILE *fout;
    // Print result to file
    if (my_id == 0)	{
	fout = fopen("OUTPUT","w");
	fclose(fout);
    }
    for (ip = 0; ip < numprocs; ip++) {
        if (my_id == ip) {
            printf("Proc %d writing to OUTPUT \n", my_id);
            fout = fopen("OUTPUT","a");

            i_offset = my_col*(N/px);
            if (my_col > (N % px))
                i_offset = i_offset + (N % px);
            else
                i_offset = i_offset + my_col;

            j_offset = my_row*(M/py);
            if (my_row > (M % py))
            j_offset = j_offset + (M % py);
            else
            j_offset = j_offset + my_row;

            for (j = 1; j < (m+1); j++) {
                for (i = 1; i < (n+1); i++) {
                    fprintf(fout,"%d \t %d \t %e\n", i+i_offset, j+j_offset, U[i][j]);
                }
            }
            if (my_left == NO_ONE)
            for (j = 1; j < (m+1); j++)
                fprintf(fout,"%d \t %d \t %e\n", 0, j+j_offset, U[0][j]);
            if (my_right == NO_ONE)
            for (j = 1; j < (m+1); j++)
                fprintf(fout,"%d \t %d \t %e\n", N+1, j+j_offset, U[n+1][j]);
            if (my_below == NO_ONE)
            for (i = 1; i < (n+1); i++)
                fprintf(fout,"%d \t %d \t %e\n", i+i_offset, 0, U[i][0]);
            if (my_above == NO_ONE)
            for (i = 1; i < (n+1); i++)
                fprintf(fout,"%d \t %d \t %e\n", i+i_offset, M+1, U[i][m+1]);
            fclose(fout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
// -----------------------------------------------------------------------------
// Compute infinity norm of values defined on a uniform 2D grid U[1..n][1..m]
// Note that index values of U range from 0..n+1, 0..m+1 but the norm is
// computed for the submesh 1..n, 1..m to exclude a layer of boundary values
double norm (int n, int m, double * V[]) {
    int i, j;
    double my_norm = 0.0;
    double nrm;
    for (i = 1; i < (n+1); i++) {
	for (j = 1; j < (m+1); j++) {
	    my_norm = fmax(my_norm,fabs(0.25*(V[i-1][j]+V[i+1][j]+V[i][j-1]+V[i][j+1])-V[i][j]));
	}
    }
    nrm = my_norm;
    // INSERT CODE HERE
    double reduced_norm;
    //printf("reduce start\n");
    MPI_Allreduce(&nrm, &reduced_norm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    //printf("reduce send\n");
    return reduced_norm;
}
// -----------------------------------------------------------------------------
// Copy mesh Vtemp[1..n][1..m] to V[1..n][1..m]
void copy_mesh (int n, int m, double * Vtemp[], double * V[]) {
    int i, j;
    for (i = 1; i < (n+1); i++) {
	for (j = 1; j < (m+1); j++) {
	    V[i][j] = Vtemp[i][j];
	}
    }
}
// -----------------------------------------------------------------------------
// Initialize left domain boundary
void initialize_left_boundary (int n, int m, double * V[]) {
    int j;
    for (j = 0; j < (m+2); j++) V[0][j] = 0.0;
}
// -----------------------------------------------------------------------------
// update left domain boundary
void update_left_boundary (int n, int m, double * V[], double * U) {
    int j;
    for (j = 0; j < (m+2); j++) V[0][j] = U[j];
}
// -----------------------------------------------------------------------------
// copy left domain boundary
void copy_left_most (int n, int m, double * V[], double * U) {
    int j;
    for (j = 0; j < (m+2); j++) U[j] = V[1][j];
}
// -----------------------------------------------------------------------------
// Initialize right domain boundary
void initialize_right_boundary (int n, int m, double * V[]) {
    int j;
    for (j = 0; j < (m+2); j++) V[n+1][j] = 0.0;
}
// -----------------------------------------------------------------------------
// Update right domain boundary
void update_right_boundary (int n, int m, double * V[], double * U) {
  int j;
  for (j = 0; j < (m+2); j++) V[n+1][j] = U[j];
}
// -----------------------------------------------------------------------------
// copy right domain boundary
void copy_right_most (int n, int m, double * V[], double * U) {
  int j;
  for (j = 0; j < (m+2); j++) U[j] = V[n][j];
}
// -----------------------------------------------------------------------------
// Initialize above domain boundary
void initialize_above_boundary (int n, int m, double * V[]) {
    int i;
    for (i = 0; i < (n+2); i++) V[i][m+1] = 1.0;
}
// -----------------------------------------------------------------------------
// Update above domain boundary
void update_above_boundary (int n, int m, double * V[], double * U) {
    int i;
    for (i = 0; i < (n+2); i++) V[i][m+1] = U[i];
}
// -----------------------------------------------------------------------------
// copy above domain boundary
void copy_above_most (int n, int m, double * V[], double * U) {
    int i;
    for (i = 0; i < (n+2); i++) U[i] = V[i][m];
}
// -----------------------------------------------------------------------------
// Initialize below domain boundary
void initialize_below_boundary (int n, int m, double * V[]) {
    int i;
    for (i = 0; i < (n+2); i++) V[i][0] = 1.0;
}
// -----------------------------------------------------------------------------
// Update below domain boundary
void update_below_boundary (int n, int m, double * V[], double* U) {
    int i;
    for (i = 0; i < (n+2); i++) V[i][0] = U[i];
}
// -----------------------------------------------------------------------------
// copy below domain boundary
void copy_below_most (int n, int m, double * V[], double * U) {
    int i;
    for (i = 0; i < (n+2); i++) U[i] = V[i][1];
}
// -----------------------------------------------------------------------------
// Update boundary values for the subdomain
// Exchange boundary values with neighbor processors
void update_boundary (int n, int m, double * U[]) {
    int i, j;
    // Processes exchange data along row of processor grid
    // my_col is even: send to right, receive from right,
    //                 send to left, receive from left
    // my_col is odd: receive from left, send to left,
    //                 receive from right, sent to right
    if ((my_col % 2) == 0) {
        if (my_right != NO_ONE) {
            // INSERT CODE HERE
            double buf_send[m+2];
            copy_right_most(n, m, U, buf_send);
            MPI_Send(&buf_send, m+2, MPI_DOUBLE, my_right, TAG, MPI_COMM_WORLD);
            double buf_recv[m+2];
            MPI_Recv(&buf_recv, m+2, MPI_DOUBLE, my_right, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            update_right_boundary(n, m, U, buf_recv);
        }
        else
            initialize_right_boundary(n, m, U);

        if (my_left != NO_ONE) {
            // INSERT CODE HERE
            double buf_send[m+2];
            copy_left_most(n, m, U, buf_send);
            MPI_Send(&buf_send, m+2, MPI_DOUBLE, my_left, TAG, MPI_COMM_WORLD);
            double buf_recv[m+2];
            MPI_Recv(&buf_recv, m+2, MPI_DOUBLE, my_left, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            update_left_boundary (n, m, U, buf_recv);
        }
        else
            initialize_left_boundary(n, m, U);
    }
    else {
    	if (my_left != NO_ONE) {
    	    // INSERT CODE HERE
          double buf_recv[m+2];
          MPI_Recv(&buf_recv, m+2, MPI_DOUBLE, my_left, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          double buf_send[m+2];
          copy_left_most(n, m, U, buf_send);
          MPI_Send(&buf_send, m+2, MPI_DOUBLE, my_left, TAG, MPI_COMM_WORLD);
          update_left_boundary (n, m, U, buf_recv);
    	}
    	else
    	    initialize_left_boundary(n, m, U);

    	if (my_right != NO_ONE) {
    	    // INSERT CODE HERE
          double buf_recv[m+2];
          MPI_Recv(&buf_recv, m+2, MPI_DOUBLE, my_right, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          double buf_send[m+2];
          copy_right_most(n, m, U, buf_send);
          MPI_Send(&buf_send, m+2, MPI_DOUBLE, my_right, TAG, MPI_COMM_WORLD);
          update_right_boundary(n, m, U, buf_recv);
    	}
    	else
    	    initialize_right_boundary(n, m, U);
    }
    // Processes exchange data along column of processor grid
    // my_row is even: send to above, receive from above,
    //                 send to below, receive from below
    // my_row is odd: receive from below, send to below,
    //                 receive from above, sent to above
    if ((my_row % 2) == 0) {
    	if (my_above != NO_ONE) {
    	    // INSERT CODE HERE
          double buf_send[n+2];
          copy_above_most(n, m, U, buf_send);
          MPI_Send(buf_send, n+2, MPI_DOUBLE, my_above, TAG, MPI_COMM_WORLD);

          double buf_recv[n+2];
          MPI_Recv(buf_recv, n+2, MPI_DOUBLE, my_above, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          update_above_boundary(n, m, U, buf_recv);
    	}
    	else
    	    initialize_above_boundary(n, m, U);

    	if (my_below != NO_ONE) {
    	    // INSERT CODE HERE
          double buf_send[n+2];
          copy_below_most(n, m, U, buf_send);
          MPI_Send(buf_send, n+2, MPI_DOUBLE, my_below, TAG, MPI_COMM_WORLD);

          double buf_recv[n+2];
          MPI_Recv(buf_recv, n+2, MPI_DOUBLE, my_below, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          update_below_boundary(n, m, U, buf_recv);
    	}
    	else
    	    initialize_below_boundary(n, m, U);
    }
    else {
    	if (my_below != NO_ONE) {
    	    // INSERT CODE HERE
          double buf_recv[n+2];
          MPI_Recv(buf_recv, n+2, MPI_DOUBLE, my_below, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          double buf_send[n+2];
          copy_below_most(n, m, U, buf_send);
          MPI_Send(buf_send, n+2, MPI_DOUBLE, my_below, TAG, MPI_COMM_WORLD);
          update_below_boundary(n, m, U, buf_recv);
    	}
    	else
    	    initialize_below_boundary(n, m, U);

    	if (my_above != NO_ONE) {
    	    // INSERT CODE HERE
          double buf_recv[n+2];
          MPI_Recv(buf_recv, n+2, MPI_DOUBLE, my_above, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          double buf_send[n+2];
          copy_above_most(n, m, U, buf_send);
          MPI_Send(buf_send, n+2, MPI_DOUBLE, my_above, TAG, MPI_COMM_WORLD);
          update_above_boundary(n, m, U, buf_recv);
    	}
    	else
    	    initialize_above_boundary(n, m, U);
    }
}
// -----------------------------------------------------------------------------
// Matvec: input = U, output = Utemp
void matvec (int n, int m, double * U[], double * Utemp[]) {
    int i, j;
    for (i = 1; i < (n+1); i++) {
	for (j = 1; j < (m+1); j++) {
	    Utemp[i][j] = 0.25*(U[i-1][j]+U[i+1][j]+U[i][j-1]+U[i][j+1]);
	}
    }
}
// -----------------------------------------------------------------------------
// Jacobi method to solve the 2D Laplace on a uniform grid
void jacobi() {
    int i, j, iter;
    double nrm;

    iter = 0;

    update_boundary(n, m, U);
    nrm = norm(n, m, U);

    if ((verbose > 0) && (my_id == 0))
	printf("Entering jacobi; iterations = %d, error norm = %e\n", iter, nrm);

    while (nrm > TOL) {
	iter++;
	matvec(n, m, U, Utemp); // matvec must be preceded by call to update_boundary
	copy_mesh(n, m, Utemp, U);
	update_boundary(n, m, U);
	nrm = norm(n, m, U);
	//if ((verbose > 0) && (my_id == 0))
	    //if ((iter % 10) == 0)
		//printf("jacobi: iterations = %d, error norm = %e\n", iter, nrm);
    }
    if ((verbose > 0) && (my_id == 0))
	printf("Exiting jacobi; iterations = %d, error norm = %32.16e\n", iter, nrm);
}
// -----------------------------------------------------------------------------
// Main Program
int main (int argc, char *argv[]) {

    int ierr;
    double start_time, total_time;

    // Set output level
    verbose = 1;

    // Initialize MPI
    ierr = MPI_Init(&argc, &argv);
    if ( ierr != MPI_SUCCESS ) {
	printf( "MPI initialization error\n" );
	exit(1);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    // Get command line input parameters
    if (argc < 5) {
    	if (my_id == 0) {
    	    printf("Use: <executable_name> <nx> <ny> <px> <py>\n");
    	    printf("for nx by ny mesh problem solved on px by py processor grid\n");
    	}
    	exit(1);
    }
    py = atoi(argv[argc-1]);
    px = atoi(argv[argc-2]);
    M = atoi(argv[argc-3]);
    N = atoi(argv[argc-4]);

    if ((verbose > 0) && (my_id == 0))
	     printf("Mesh (NxM): %d x %d\t Processor grid (px x py): %d x %d\n", N, M, px, py);

    // Check input parameters
    ierr = 0;
    if (my_id == 0) {
    	if (px*py != numprocs) {
    	    printf("Procs: %d not equal to (px * py): %d x %d ... aborting\n", numprocs, px, py);
    	    ierr = 1;
    	}
    	if ((N < 1) || (M < 1)) {
    	    printf("Error: m, n must be > 0 (n=%d, m=%d) ... aborting\n", N, M);
    	    ierr = 1;
    	}
    	if ((N < px) || (M < py)) {
    	    printf("Error: (px > N or py > M) ... aborting\n");
    	    ierr = 1;
    	}
    }
    MPI_Bcast(&ierr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ierr != 0) MPI_Finalize();

    // Initialize process coordinates and determine neighbors
    initialize_process_info();

    // Initialize local array size
    // Here, N % px is the remainder
    n = N/px; if (my_col < (N % px)) n = n+1;
    m = M/py; if (my_row < (M % py)) m = m+1;

    if (verbose > 0) {
    	printf("Proc-grid (%1d x %1d), my_id = %d, (row,col)=(%d,%d)",
    		px, py, my_id, my_row, my_col);
    	printf(" neighbors:(%d, %d, %d, %d)", my_left, my_right, my_above, my_below);
    	printf(" Mesh Global:%1d x %1d, Local:%1d x %1d\n", N, M, n, m);
    }

    // Allocate mpi buffer
    buf = allocate_mpi_buffer (n, m);

    // Allocate and initialize meshes
    U = allocate_2d_array(n+2, m+2);
    initialize_mesh(n+2, m+2, U);
    Utemp = allocate_2d_array(n+2, m+2);
    initialize_mesh(n+2, m+2, Utemp);

    // Jacobi iterative solver
    start_time = MPI_Wtime();
    jacobi();
    total_time = MPI_Wtime()-start_time;

    if (my_id == 0)
	   printf(" Mesh Global:%1d x %1d, Local:%1d x %1d, Total time = %f\n", N, M, n, m, total_time);

    output_solution();

    // Free memory
    free_2d_array(n+2, m+2, U);
    free_2d_array(n+2, m+2, Utemp);

    MPI_Finalize();
    return 0;
}
// =============================================================================
