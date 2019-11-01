/**

    Author:  	Athanasios Kirmizis
    Dept.:   	EE AUTH
    AEM :    	8835
    Course:  	Parallel And Distributed Systems
	Assignment: #1
	Season:  	2019 - 2020
    E-mail : 	athakirm@ece.auth.gr
    Prof:    	Nikolaos Pitsianis | Dimitrios Floros

    Build a Vantage Point Tree using Cilk

**/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <cilk/cilk.h>
#include "vptree.h"

#define SWAP(x, y) { double temp = x; x = y; y = temp; }
#define MAX_ALIVE 16
#define SEQUENTIAL_THRESHOLD 1000

/**Global Variables**/

pthread_mutex_t mux;
int aliveThreads;

/**
* Partition using Lomuto partition scheme
*
* @param a: Array to perform partition on
* @param left: Most left element index
* @param right: Most right element index
* @param pivotIndex: Index to the pivot element
* @returns Index to the new pivot element
*
**/
int partition(double a[], int left, int right, int pivotIndex)
{
	//Pick pivotIndex as pivot from the array
	double pivot = a[pivotIndex];

	//Move pivot to end
	SWAP(a[pivotIndex], a[right]);

	//Elements less than pivot will be pushed to the left of pIndex
	//Elements more than pivot will be pushed to the right of pIndex
	//Equal elements can go either way
	int pIndex = left;
	int i;

	//Each time we finds an element less than or equal to pivot, pIndex
	//is incremented and that element would be placed before the pivot.
	for (i = left; i < right; i++)
	{
		if (a[i] <= pivot)
		{
			SWAP(a[i], a[pIndex]);
			pIndex++;
		}
	}

	//Move pivot to its final place
	SWAP(a[pIndex], a[right]);

	//Return pIndex (index of pivot element)
	return pIndex;
}

/**
* Quickselect algorithm
*
* @param A: Array to perform quickselect on
* @param left: Most left element index
* @param right: Most right element index
* @param k: The number of element (going from smaller to larger) to pick from the array
* @returns The kth smaller element of the array
*
**/
double quickselect(double A[], int left, int right, int k)
{
	//If the array contains only one element, return that element
	if (left == right)
		return A[left];

	//Select a pivotIndex between left and right
	int pivotIndex = left + rand() % (right - left + 1);

	pivotIndex = partition(A, left, right, pivotIndex);

	//The pivot is in its final sorted position
	if (k == pivotIndex)
		return A[k];

	//If k is less than the pivot index
	else if (k < pivotIndex)
		return quickselect(A, left, pivotIndex - 1, k);

	//If k is more than the pivot index
	else
		return quickselect(A, pivotIndex + 1, right, k);
}


/**
* Recursevily builds the vantage point tree
*
* @param n: Number of data points (rows of X)
* @param dim: Number of dimensions (columns of X)
* @param x: Input data points
* @param idxs: Array of indexes to the original positions of the set of points
* @returns A pointer to the produced tree of struct vptree
*
**/

struct vptree * buildvp(double *X, int n, int d)
{
	//Initialize some values
	aliveThreads = 0;
	pthread_mutex_init (&mux, NULL);
	
	//Create a node and call the recursive function
    	struct vptree *T = malloc(sizeof(struct vptree));

	int *idx = malloc(n*sizeof(int));
	for (int i=0; i<n; i++) {

		idx[i] = i;
	}
	
	T = vpt(n-1, d, X, idx);

	return T;
}

/*-------- Here is the "nested" function that will run recursively ---------- */

struct vptree * vpt(int n, int dim, double *x, int *idxs)
{
	struct vptree *t = malloc(sizeof(struct vptree));
	t->vp = (double *)malloc(dim*sizeof(double));

	if(n < 0){
		
		return NULL;
	}
	//If leaf node, return the node
	else if(n == 0) {
		
		for(int i=0; i<dim; i++) {

			t->vp[i] = x[i];
		}
		t->idx = idxs[0];
	
		return t;
	}
	//Else keep building the tree from this node on
	else {

		for(int i=0; i<dim; i++) {

			t->vp[i] = x[n*dim + i];
		}
		t->idx = idxs[n];

		///!!///Do the work for the distances here///!!///

		//Find the euclidean distance of this point to all the other points of this sub-tree
		double *d = (double *)malloc(n*sizeof(double));
		double *tempD = (double *)malloc(n*sizeof(double));
		
		//If there is too little work, switch to sequential processing
		if((n < SEQUENTIAL_THRESHOLD) || (aliveThreads >= MAX_ALIVE/2)){
			
			for(int i=0; i<n; i++) {

				d[i] = 0;
				for(int j=0; j<dim; j++) {

					d[i] = d[i] + (double) pow((x[i*dim + j] - x[n*dim + j]),2);
				}
				d[i] = (double) sqrt(d[i]);
			}
		}
		//If there is enough work to be done, parallelize the procedure

		else{

			//Threshold the number of threads here
			int threadCap;
			int threadDivider;
		
			threadCap = MAX_ALIVE - aliveThreads;
			if(threadCap == 0) threadCap = 2;			
			threadDivider = n/threadCap;
			
			#pragma grainsize = threadDivider
			cilk_for(int i=0; i<n; i++){

				d[i] = 0;
				for(int j=0; j<dim; j++) {

					d[i] = d[i] + (double) pow((x[i*dim + j] - x[n*dim + j]),2);
				}
				d[i] = (double) sqrt(d[i]);
			}
			
		}
		
		memcpy(tempD, d, n*sizeof(double));

		//Calculate the median distance
		int k = (int)(n/2.0 + 0.5);
		double medianDistance = quickselect(tempD, 0, n-1, k-1);
		t->md = medianDistance;

		//Keep a new array for the inner sub-tree
		double *y = (double *)malloc(k*dim*sizeof(double));
		int *idxsIn = malloc(k*sizeof(int));
		int counterIn = 0;

		//Keep a new array for the outer sub-tree
		double *z = (double *)malloc((k-(n%2))*dim*sizeof(double));
		int *idxsOut = malloc((k-(n%2))*sizeof(int));
		int counterOut = 0;

		//Build the two new sub-arrays of points
		for(int i=0; i<n; i++){

			if(d[i] <= medianDistance) {

				for(int j=0; j<dim; j++) {

					y[counterIn*dim+j] = x[i*dim + j];
				}
				idxsIn[counterIn] = idxs[i];
				counterIn++;
			}
			else {

				for(int j=0; j<dim; j++) {

					z[counterOut*dim+j] = x[i*dim + j];
				}
				idxsOut[counterOut] = idxs[i];
				counterOut++;
			}
		}
		
		//We don't need the initial whole array anymore so free it
		//Commented out to work with e-learning tester
		//free(x);

		///!!///Do the work for the inner and the outer sub-trees here///!!///
			
		//If there is too little work, switch to sequential processing	
		if((n < SEQUENTIAL_THRESHOLD) || (aliveThreads >= MAX_ALIVE)){

			//Do the recursions
			struct vptree *in = malloc(sizeof(struct vptree));
			in = vpt(k-1, dim, y, idxsIn);
			t->inner = in;
			struct vptree *out = malloc(sizeof(struct vptree));
			out = vpt(k-(n%2)-1, dim, z, idxsOut);
			t->outer = out;
		}
		//If there is enough work to be done, parallelize the procedure
		//One thread will be created for the inner sub-tree
		//And the outer sub-tree will be calculated in the master thread to save time and resources 

		else {
			
			pthread_mutex_lock (&mux);
			aliveThreads += 2;
			pthread_mutex_unlock (&mux);
			
			//Do the recursions		
			struct vptree *in = cilk_spawn vpt(k-1, dim, y, idxsIn); 
			struct vptree *out = cilk_spawn vpt(k-(n%2)-1, dim, z, idxsOut); 
			
			cilk_sync;

			pthread_mutex_lock (&mux);
			aliveThreads -= 2;
			pthread_mutex_unlock (&mux);	

			t->inner = in;
			t->outer = out;
			
		} 

		return t;

	}
}

/*-------- List of accessors ---------- */

struct vptree * getInner(struct vptree * T)
{

	return T->inner;
}

struct vptree * getOuter(struct vptree * T)
{

	return T->outer;
}

double getMD(struct vptree * T)
{

	return T->md;
}

double * getVP(struct vptree * T)
{

	return T->vp;
}

int getIDX(struct vptree * T)
{

	return T->idx;
}
