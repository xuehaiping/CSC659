#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

//print an m*n matrix
void print(int m, int n, double A[][n]) {
  int i,j;
    for (i = 0; i < m; i++)
    {
      for (j = 0; j < n; j++)
         printf("%10.4e ", A[i][j]);
      printf("\n");
    }
}

double norm(int n, double x[][n])
{
    int i,j;
    double sum = 0.0;
    for (i = 0; i < n; i++)
      for(j=0; j < n; j++)
        sum = sum + x[i][j]*x[i][j];
    return sqrt(sum);
}

//matrix minus with respect to two matrix
void multiply(int m, int n, double m1[][n], int p, int q, double m2[][q], double res[][q])
{
    int c, d, k;
    double sum = 0;
    for (c = 0; c < m; c++) {
      for (d = 0; d < q; d++) {
        for (k = 0; k < p; k++) {
          sum = sum + m1[c][k]*m2[k][d];
        }
        res[c][d] = sum;
        sum = 0;
      }
    }
}

//matrix minus with respect to two n*n matrix
void minus(int n, double m1[][n], double m2[][n]) {
  int i, j;
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      m1[i][j] = m1[i][j] - m2[i][j];
    }
  }
}

//initialize the matrix with random value
void initialize_matrix(int n, double A[][n])
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i][j] = (double)rand()/((double)RAND_MAX + 1);
}

//cut a matrix to triangular matrix
void triu(int n, double A[][n]) {
  int i, j;
  for(i=0; i<n; i++) {
    for(j=0; j<i; j++) {
        A[i][j] = 0;
    }
  }
}

//add absolute sum to diagnal elements
void absSumDiag(int n, double A[][n]) {
  int i, j;
  double sum[n];
  for(i=0; i<n; i++) {
    sum[i] = 0;
    for(j=0; j<n; j++) {
        sum[i] += fabs(A[j][i]);
    }
  }

  for(i=0; i<n; i++) {
    A[i][i] += sum[i];
  }
}

//copy R11, R22 matrix
void copyM(int n, double A[][n], int start, int size, double res[][size]) {
  int i,j;
  for(i=0; i<size; i++) {
    for(j=0; j<size; j++)
      res[i][j] = A[i+start][j+start];
  }
}

//assign values to matrix
void assginM(int size, int startX, int startY, double A[][size], int p, int q, double set[][q]) {
  int i,j;
  for(i=0; i<p; i++) {
    for(j=0; j<q; j++)
      A[startX+i][startY+j] = set[i][j];
  }
}

//inverse an upper triangular matrix
void inverseUp(int n, double A[][n], double res[][n]) {
  double cp[n][2*n];
  int i,j;
  //intialize the matrix
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++)
      if(i==j)
        cp[i][j+n]=1;
      else
        cp[i][j+n]=0;
  }

  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      cp[i][j] = A[i][j];
    }
  }

  //normalize the cp matirx
  for(i=0; i<n; i++) {
    double divider = cp[i][i];
    for(j=0; j<2*n; j++) {
      cp[i][j] = cp[i][j]/divider;
    }
  }

  //back-substitution to find the inverse of the matrix
  int p, r, c;
  for (p=n-1;p>0;p--) {
        for (r=p-1;r>=0;r--) {
            double multiple = cp[r][p] / cp[p][p];
            for (c=p-1;c<2*n;c++) {
                cp[r][c] = cp[r][c] - cp[p][c]*multiple;
            }
        }
    }

  //copy the result to the result matrix
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      res[i][j] = cp[i][j+n];
    }
  }
}

//computer upper traingular matrix's inverse
void computerInverse(int n, double A[][n]) {
  //printf("current size %d\n", n);
  //compute the inverse when submatrix's size
  if(n<16) {
    double res[n][n];
    inverseUp(n, A, res);
    assginM(n, 0, 0, A, n, n, res);
  }
  else {
    //compute R11 and R22's size
    int n1 = round(n/(double)(2));
    double R11[n1][n1];
    copyM(n, A, 0, n1, R11);
    double R22[n-n1][n-n1];
    copyM(n, A, n1, n-n1, R22);

    computerInverse(n1, R11);

    double sizeR22 = n-n1;
    computerInverse(sizeR22, R22);

    //copy R12
    double R12[n1][n-n1];
    int i, j;
    //copy R12
    for(i=0; i<n1; i++) {
      for(j=0; j<n-n1; j++)
        R12[i][j] = A[i][n1+j];
    }

    //set R12
    double resMul[n1][n-n1];
    multiply(n1, n1, R11, n1, n-n1, R12, resMul);
    multiply(n1, n-n1, resMul, n-n1, n-n1, R22, R12);

    for(i=0; i<n1; i++) {
      for(j=0; j<n-n1; j++) {
        R12[i][j] *= -1.0;
      }
    }

    //set R12 to A
    assginM(n, 0, n1, A, n1, n-n1, R12);
    //set R11 to A
    assginM(n, 0, 0, A, n1, n1, R11);
    //set R22 to A
    assginM(n, n1, n1, A, n-n1, n-n1, R22);
  }
}

// create an n*n indentity matrix
void IdentityM(int n, double A[][n]) {
  int i,j;
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      if(i!=j)
        A[i][j] = 0;
      else
        A[i][j] = 1;
    }
  }
}



int main(int argc, char **argv)
{
  int n;
  n = atoi(argv[1]);

  //abort when n is negative
  if (n <= 0) {
      printf("n must be a positive integer ... aborting\n");
      exit(0);
  }

  //initialize the matrix with random number
  double m1[n][n];
  initialize_matrix(n, m1);
  //make the diagnal not equal to 0
  absSumDiag(n, m1);
  //traform to a upper triangular matrix
  triu(n, m1);

  //copy the matrix for inverse
  double reverse[n][n];
  copyM(n, m1, 0, n, reverse);

  //compute the time of this routine
  double start = omp_get_wtime();
  //inverse the matrix
  computerInverse(n, reverse);
  double elapsed_time = omp_get_wtime() - start;

  //check the result
  double mulRes[n][n];
  multiply(n, n, m1, n, n, reverse, mulRes);
  //build Identity matrix
  double id[n][n];
  IdentityM(n, id);

  //check the correctness of the answer
  minus(n, mulRes, id);
  printf("Result of norm: %10.4e\n", norm(n, mulRes));
  printf("Time used for computerInverse routine: %10.4e\n", elapsed_time);
  // printf("Result matrix is \n");
  // print(n, n, m1);
}
