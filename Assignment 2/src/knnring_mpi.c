#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <cblas.h>
#include "mpi.h"
#include "../inc/knnring.h"
#include "../inc/auxiliary_functions.h"

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

			A[i] = A[i] + pow(X[i*d + j],2);	
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
		for(int j=0; j<k; j++) {

			res.ndist[i*k + j] = quickselect(tempDist,tempIdx,0,k-1,j);
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

//! Compute distributed all-kNN of points in X
/*!
\param X Data points [n-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/

knnresult distrAllkNN(double * X, int n, int d, int k) {
	
	//!--- At the beginning, some preparation of the MPI environment ---! 

	//Some MPI necessary values
	int nproc, pid;
	MPI_Status recvstatus, sendstat;
	MPI_Request sendreq, recvreq;
	int tag = 50;

	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
	int cap = n;

	//Initialize and setup result structure to return
	knnresult finalProcessResult;
	finalProcessResult.m = cap;
	finalProcessResult.k = k;
	finalProcessResult.nidx = malloc(cap*k*sizeof(int));
	finalProcessResult.ndist = malloc(cap*k*sizeof(double));

	//Set the distances to MAX for comparison purposes further on
	for(int i=0; i<cap; i++) {
		for(int j=0; j<k; j++) {
			
			finalProcessResult.ndist[i*k + j] = DBL_MAX;
			finalProcessResult.nidx[i*k + j] = -1;
		}
	}
	
	//!--- Each process then calculates the knnresult between its set of points X ---! 
	//!--- and another block of points Z					      				  ---!
	
	knnresult tempProcessResult;
	tag += pid;
	
	double *Z = malloc(cap*d*sizeof(double));
	double *tempZ = malloc(cap*d*sizeof(double));

	//For the first loop, set Z to be the same block of points as X
	memcpy(Z, X, cap*d*sizeof(double));
	int buffLen = cap*d;
	
	//For each block of points
	for(int p=0; p<nproc; p++) {

		//Begin sending your block of points Z to the next process
		MPI_Isend(Z, buffLen, MPI_DOUBLE, (pid+1)%nproc, tag, MPI_COMM_WORLD, &sendreq);

		//Declare your wish to receive the new block of points Z from previous process
		MPI_Irecv(tempZ, cap*d, MPI_DOUBLE, (nproc+pid-1)%nproc, MPI_ANY_TAG, MPI_COMM_WORLD, &recvreq);

		//Calculate the kNN for Z and X
		tempProcessResult = kNN(Z, X, cap, cap, d, k);

		//Given that result, update the final result accordingly
		for(int i=0; i<cap; i++) {
	
			//Temp arrays for picking the k smallest of both tempResult and finalResult
			double *tempDist = malloc(2*k*sizeof(double));
			int *tempIdx = malloc(2*k*sizeof(int));
			
			//Adjust the points indexing with an offset
			int indexOffset = (nproc + pid - p - 1)%nproc;

			//Temp arrays are basically previous and current results concatenated so they can be sorted at once
			for(int j=0; j<k; j++) {
			
				tempIdx[j] = finalProcessResult.nidx[i*k + j]; 
				tempDist[j] = finalProcessResult.ndist[i*k + j]; 
			}

			for(int j=k; j<2*k; j++) {
				
				tempIdx[j] = tempProcessResult.nidx[i*k + j - k] + cap*indexOffset;
				tempDist[j] = tempProcessResult.ndist[i*k + j - k];
			}

			doubleBubbleSort(tempDist, tempIdx, 2*k);
			
			//Update the ndist and nidx tables with the k smallest distances
			for(int j=0; j<k; j++) {
				
				finalProcessResult.ndist[i*k + j] = tempDist[j];
				finalProcessResult.nidx[i*k + j] = tempIdx[j];
			}
	
			free(tempDist);
			free(tempIdx);
		}

		MPI_Wait(&sendreq, &sendstat);
		MPI_Wait(&recvreq, &recvstatus);

		MPI_Get_count(&recvstatus, MPI_DOUBLE, &buffLen);

		//Update block of points Z
		memcpy(Z, tempZ, cap*d*sizeof(double));		
	}

	free(Z);
	free(tempZ);

	//!--- Before finishing, also show the global minimum and maximum distances found ---!
	//!--- on the final kNN distances using MPI reductions							  ---!

	double process_max = -1;
	double process_min = DBL_MAX;
	double global_max = 0;
	double global_min = 0;

	//In the end the results for each point are sorted
	for(int i=0; i<cap; i++) {
		
		//So simply check on the second (to ignore distance from self which is always 0)
		if(process_min > finalProcessResult.ndist[i*k + 1]) {

			process_min = finalProcessResult.ndist[i*k + 1];
		}
		//And simply check on the last element
		if(process_max < finalProcessResult.ndist[i*k + k-1]) {

			process_max = finalProcessResult.ndist[i*k + k -1];
		}

	}

	MPI_Reduce(&process_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&process_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	
	if(pid == 0) {
		
		printf("\nGlobal minimum distance found on the k=%d nearest neighbors: %f\n", k, global_min);
		printf("Global maximum distance found on the k=%d nearest neighbors: %f\n\n", k, global_max);
	}

	//!--- Return result ---!

	return finalProcessResult;
}
