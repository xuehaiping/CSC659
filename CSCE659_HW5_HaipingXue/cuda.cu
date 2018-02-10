#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda_runtime.h"

void printVect(float* xi, int l) {
  int i;
  for(i=0; i<l; i++) {
    printf("%8.10f ", xi[i]);
  }
  printf("\n");
}

void printMat(float* mat, int n) {
  int row, col;
  for(row=0; row<n; row++) {
    for(col=0; col<n; col++) {
      printf("%8.10f ", mat[row*n+col]);
    }
    printf("\n");
  }
}

void computePoints(float* xi, float* yi, float h, int m) {
  //compute all points
  int i, j, idx = 0;
  for(i=1; i<=m; i++) {
    for(j=1; j<=m; j++) {
      xi[idx] = i*h;
      yi[idx] = j*h;
      idx = idx+1;
    }
  }
}

float computeRes(float* kt, float* df, int n) {
  int i;
  float sum = 0;
  for(i=0; i<n; i++) {
    sum += kt[i]*df[i];
  }
  return sum;
}

__global__ void computeKT(float* xi, float* yi, float* kt, int n, float xp, float yp) {
  int j = blockIdx.x;
  kt[j] = exp(-1.0*((xp-xi[j])*(xp-xi[j]) + (yp-yi[j])*(yp-yi[j])));
}

__global__ void computeK(float* xi, float* yi, float* a, int n) {
  int numRows = n/blockDim.x;
  //xi,yi 1*4 0,4
  for(int k=0;k<numRows;k++){
    int i = threadIdx.x*numRows+k;
    if(i>=n)
      break;
    for(int j=0;j<n;j++){
      int idx = i*n + j;
      a[idx] = exp(-1.0*((xi[i]-xi[j])*(xi[i]-xi[j]) + (yi[i]-yi[j])*(yi[i]-yi[j])));
      if(i==j) {
        a[idx] = a[idx] + 0.01;
      }
    }
  }
}

void computeF(float* xi, float* yi, float* f, int n) {
  int i;
  for(i=0; i<n; i++) {
    float d = (float)0.1 * ((((float)rand()) / (float)RAND_MAX)-0.5);
    f[i] = (float)1.0 - ((xi[i]-0.5) * (xi[i]-0.5) + (yi[i]-0.5) * (yi[i]-0.5)) + d;
  }
}


__global__ void LU(float *a, int n, int blockSize) {
    // Normal parallel
    // int k, y;
    // int numRows = n/blockDim.x;
    // int start = threadIdx.x;
    //
    // for(k=0;k<n-1;k++){
    //
    //   for(y=0;y<numRows;y++){
    //     int i = threadIdx.x*numRows + y;
    //     if(i>k&&i<n){
    //       int Aik = i*n + k, Akk = n*k + k;
    //       a[Aik] = a[Aik]/a[Akk];
    //     }
    //   }
    //   __syncthreads();
    //
    //   int l, z;
    //   for(l=0; l<numRows; l++) {
    //     int i = threadIdx.x*numRows + l;
    //
    //     if(i>k && i<n) {
    //       for(z=k+1; z<n; z++) {
    //         int Aiz = i*n + z, Aik = i*n + k, Akz = k*n + z;
    //         // printf("i= %d z= %d\n", i, z);
    //         // printf("Aik= %d Akz= %d Aiz=%d\n", Aik, Akz, Aiz);
    //         // printf("threadIdx= %d\n", threadIdx.x);
    //         a[Aiz] = a[Aiz] - a[Aik]*a[Akz];
    //       }
    //     }
    //   }
    //    __syncthreads();
    // }

    // cyclic row partion method
    int k, y;
    int numRows = n/blockDim.x;
    int start = threadIdx.x;

    for(k=0;k<n-1;k++){

      for(y=0;y<numRows;y++){
        int i = y*blockDim.x+start;
        if(i>k&&i<n){
          int Aik = i*n + k, Akk = n*k + k;
          a[Aik] = a[Aik]/a[Akk];
        }
      }

      //__syncthreads();

      int l, z;
      for(l=0; l<numRows; l++) {
        int i = l*blockDim.x+start;

        if(i>k && i<n) {
          for(z=k+1; z<n; z++) {
            int Aiz = i*n + z, Aik = i*n + k, Akz = k*n + z;
            a[Aiz] = a[Aiz] - a[Aik]*a[Akz];
          }
        }
      }
       __syncthreads();
    }
}

//solve L
__global__ void solveL(float *x, float *L, int n) {
  int i, k;
  int numRows = n/blockDim.x;

  for (i = 1; i < n; i++) {
    for(k=0;k<numRows; k++){
      int j = threadIdx.x*numRows + k;
      int LIdx = j*n + i - 1;
      if(j>=i&&j<n)
        x[j] = x[j] - L[LIdx]*x[i-1];
    }
    __syncthreads();
  }
}

//solve U
__global__ void solveU(float *x, float *U, int n) {
  int i, k;
  int numRows = n/blockDim.x;

  for (i = n-1; i > 0; i--) {
    if(threadIdx.x==0)
      x[i] = x[i]/U[i*n+i];
    __syncthreads();

    for(k=0;k<numRows;k++){
      int j = threadIdx.x*numRows + k;
      int UIdx = j*n + i;
      if(j>=0&&j<i)
        x[j] = x[j] - U[UIdx]*x[i];
    }
      __syncthreads();
  }
  if(threadIdx.x==0)
      x[0] = x[0]/U[0];
}


int main(int argc, char **argv)
{
  int m, n, size, matSize, blockSize;
  float xp, yp;
  /* Check input parameters */
  m = atoi(argv[1]);
  xp = atof(argv[2]);
  yp = atof(argv[3]);
  blockSize = atoi(argv[4]);
  n = m*m;
  size = n*sizeof(float);
  matSize = n*n*sizeof(float);

  float* xi = (float*)malloc(size);
  float* yi = (float*)malloc(size);
  float* f = (float*)malloc(size);
  //float* K = (float*)malloc(matSize);
  float h = ((float)1.0)/(float)(m+1);

  //Initialize points and f
  computePoints(xi, yi, h, m);
  computeF(xi, yi, f, n);
  cudaSetDevice(0);
  // allocate vector in device memory
  float* dXi;
  cudaMalloc(&dXi, size);
  float* dYi;
  cudaMalloc(&dYi, size);
  float* dK;
  cudaMalloc(&dK, matSize);
  //copy f to device
  float* df;
  cudaMalloc(&df, size);
  cudaMemcpy(df, f, size, cudaMemcpyHostToDevice);

  //copy vectors to device
  cudaMemcpy(dXi, xi, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dYi, yi, size, cudaMemcpyHostToDevice);

  // Invoke kernel
  computeK<<<1, blockSize>>>(dXi, dYi, dK, n);

  //NOTE: K for debug
  //cudaMemcpy(K, dK, matSize, cudaMemcpyDeviceToHost);

  // FILE * fp;
  // /* open the file for writing*/
  // fp = fopen ("origin_Matrix","w+");
  // int row, col;
  // /* write 10 lines of text into the file stream*/
  // for(row=0; row<n; row++) {
  //   for(col=0; col<n; col++) {
  //     fprintf (fp, "%8.10f ", K[row*n+col]);
  //   }
  //   fprintf(fp, "\n");
  // }
  // fclose(fp);
  //printMat(K,n);

  printf("Start computing LU\n");

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );


  LU<<<1, blockSize>>>(dK, n, blockSize);

  cudaEventRecord( stop, 0);
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );

  printf("Time for LU factors routines:  %4.10f s \n", time/1000);

  //NOTE: K for debug
  //cudaMemcpy(K, dK, matSize, cudaMemcpyDeviceToHost);
  /* open the file for writing*/
  // fp = fopen ("LU_Matrix","w+");
  // /* write 10 lines of text into the file stream*/
  // for(row=0; row<n; row++) {
  //   for(col=0; col<n; col++) {
  //     fprintf (fp, "%8.10f ", K[row*n+col]);
  //   }
  //   fprintf(fp, "\n");
  // }
  //
  // fclose(fp);

  // /* open the file for writing*/
  // fp = fopen ("fVec","w+");
  // /* write 10 lines of text into the file stream*/
  // for(row=0; row<n; row++) {
  //   fprintf (fp, "%8.10f ", f[row]);
  // }
  cudaEvent_t start1, stop1;
  float time1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cudaEventRecord( start, 0 );

  solveL<<<1,blockSize>>>(df, dK, n);

  solveU<<<1,blockSize>>>(df, dK, n);

  cudaEventRecord( stop1, 0);
  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &time1, start1, stop1);
  cudaEventDestroy( start1 );
  cudaEventDestroy( stop1 );

  printf("Time for solver routine is:  %4.10f s \n", time1/1000);

  //NOTE: K for debug
  //cudaMemcpy(K, dK, matSize, cudaMemcpyDeviceToHost);
  /* open the file for writing*/
  // fp = fopen ("LU_Matrix","w+");
  // /* write 10 lines of text into the file stream*/
  // for(row=0; row<n; row++) {
  //   for(col=0; col<n; col++) {
  //     fprintf (fp, "%8.10f ", K[row*n+col]);
  //   }
  //   fprintf(fp, "\n");
  // }

  float* res = (float*)malloc(size);
  cudaMemcpy(res, df, size, cudaMemcpyDeviceToHost);

  /* open the file for writing*/
  // fp = fopen ("fRes","w+");
  // /* write 10 lines of text into the file stream*/
  // for(row=0; row<n; row++) {
  //   fprintf (fp, "%8.10f ", res[row]);
  // }

  //create the kt on the device
  float* DkTrans;
  cudaMalloc(&DkTrans, size);
  //computer kt
  computeKT<<<n,1>>>(dXi, dYi, DkTrans, n, xp, yp);
  float* kt = (float*)malloc(size);
  cudaMemcpy(kt, DkTrans, size, cudaMemcpyDeviceToHost);

  //compute result
  float result;
  result = computeRes(kt, res, n);
  printf(" result is: %8.10f\n", result);
  //free all memory
  free(xi);
  free(yi);
  free(f);
  //free(K);
  free(res);
  free(kt);
  cudaFree(dXi);
  cudaFree(dYi);
  cudaFree(dK);
  cudaFree(df);
  cudaFree(DkTrans);
}
