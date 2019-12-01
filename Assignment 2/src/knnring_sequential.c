#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include "../inc/knnring.h"
#include "../inc/auxiliary_functions.h"
#include <cblas.h>

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!
\param X Corpus data points [n-by-d]
\param Y Query data points [m-by-d]
\param n Number of corpus points [scalar]
\param m Number of query points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k)
{
	
	//Initialize and setup result structure to return
	knnresult res;
	res.nidx = malloc(m*k*sizeof(int));
	res.ndist = malloc(m*k*sizeof(double));
	res.m = m;
	res.k = k;

	//Arrays used to calculate Euclidean Matrix
	double *A = (double *)malloc(n*sizeof(double));
	double *B = (double *)malloc(m*sizeof(double));
	double *C = (double *)malloc(m*n*sizeof(double));
	double *D = (double *)malloc(m*n*sizeof(double));

	//Calculate sum(X.^2,2)
	for(int i=0; i<n; i++) {

		A[i] = 0;
		for(int j=0; j<d; j++) {

			A[i] = A[i] + X[i*d + j]*X[i*d + j];	
		}
	}

	//Calculate sum(Y.^2,2).'
	for(int i=0; i<m; i++) {

		B[i] = 0;
		for(int j=0; j<d; j++) {

			B[i] = B[i] + Y[i*d + j]*Y[i*d + j];	
		}
	}
	
	//Use BLAS to calculate 2*X*Y.'
	double alpha = 2;
	double beta = 0;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, alpha, X, d, Y, d, beta, C, m);

	//Calculate final result
	for(int i=0; i<n; i++) {
		for(int j=0; j<m; j++) {
			
			//This is to avoid negative values due to rounding errors
			if(A[i] - C[i*m + j] + B[j] < 0) {

				D[i*m + j] = 0;
			}
			else {
				
				D[i*m + j] = sqrt(A[i] - C[i*m + j] + B[j]);
			}
		}
	}
	
	//Temp arrays for picking the k-th smallest
	double *tempDist = malloc(n*sizeof(double));
	int *tempIdx = malloc(n*sizeof(int));
	
	//For each point of Y
	for(int i=0; i<m; i++) {
		
		//Build the indices of X and distances from X points vector
		for(int j=0; j<n; j++) {
			
			tempIdx[j]=j;
			tempDist[j] = D[i + j*m];
		}

		//Perform an initial quickselect for k
		res.ndist[i*k + k-1] = quickselect(tempDist,tempIdx,0,n-1,k-1);
		res.nidx[i*k + k-1] = tempIdx[k-1];

		//Then perform another k-1 quickselects but search only to the left of previous found element
		for(int j=k-2; j>=0; j--) {
			
			res.ndist[i*k + j] = quickselect(tempDist,tempIdx,0,j,j);
			res.nidx[i*k + j] = tempIdx[j];
		}
	}

	//Free unnecessary space
	free(D);
	free(A);
	free(B);
	free(C);
	free(tempDist);
	free(tempIdx);
	
	return res;
}
