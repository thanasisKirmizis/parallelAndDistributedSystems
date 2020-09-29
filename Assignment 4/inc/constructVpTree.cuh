/**

    Author:  	Athanasios Kirmizis
    Dept.:   	EE AUTH
    AEM :    	8835
    Course:  	Parallel And Distributed Systems
	Assignment: #4
	Season:  	2019 - 2020
    E-mail : 	athakirm@ece.auth.gr
    Prof:    	Nikolaos Pitsianis | Dimitrios Floros

    Functions to Build a Vantage Point Tree Using CUDA for its Construction

**/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include "vptree.h"

#define SWAP(x, y) { double temp = x; x = y; y = temp; }

__device__ int found = 0;

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

__device__ int partition(double a[], int left, int right, int pivotIndex)
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

__device__ double quickselect(double A[], int left, int right, int k)
{
	//If the array contains only one element, return that element
	if (left == right)
		return A[left];

	//Random does not exist in GPU code, so simply take RAND_MAX
	int rand1 = RAND_MAX;

	//Select a pivotIndex between left and right
	int pivotIndex = left + rand1 % (right - left + 1);

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

/*-------- Here is the "device" helper function that calculates a node's info ---------- */

__device__ void vpt(int initN, int dim, double *x, int* splitOfPoints, double* d, int id, int depth, int maxDepth, struct vptree *tree)
{
	//Split the whole node's work to multiple blocks with multiple threads
	int threadId = blockIdx.y*blockDim.x + threadIdx.x;

	//Declare some needed values of the node to be shared between all threads
	__shared__ int tIdx;
	__shared__ int tInnerSplitPtr;
	__shared__ int numThreads;
	struct vptree t;

	//Define the parent node ID (if this is not the root node)
	int parentNodeId = 0;
	if(id != 0) parentNodeId = (int)((id - 1)/2.0);

	//Define the number of points in this subTree and k
	int n;
	if(id == 0) n = initN;
	else if(id%2 == 0) n = (int)( (tree[parentNodeId].numOfPointsInSubtree-1)/2.0 );
	else if(id%2 == 1) n = (int)( (tree[parentNodeId].numOfPointsInSubtree)/2.0 );
	int k = (int)(n/2.0 + 0.5);

	//Make all threads except for master thread to return if this is a leaf or a null node
	if(n == 1 && threadId != 0) return;
	if(n == 0 && threadId != 0) return;

	//The first thread of each block that handle this node will do some common work
	if(threadIdx.x == 0) {
		
		numThreads = gridDim.y*blockDim.x;

		//If child of a leaf node, add a dummy node and return
		if(n == 0) {
			
			t.idx = -1;
			tree[id] = t;
			return;
		}

		//Fill the struct with some info
		t.numOfPointsInSubtree = n;
		t.vp = (double *)malloc(dim*sizeof(double));
		
		//Handle the root node
		if(id == 0) {

			//Assign info for the root node point
			t.idx = n - 1;
			tIdx = t.idx;

			for(int i=0; i<dim; i++) {
				
				t.vp[i] = x[(t.idx)*dim + i];
			}

			t.innerSplitPtr = 0;
			t.outerSplitPtr = n-1;
			tInnerSplitPtr = t.innerSplitPtr;

		}
		//A left node (with nodeId an odd number)
		else if(id%2 == 1) {

			t.innerSplitPtr = tree[parentNodeId].innerSplitPtr;
			t.outerSplitPtr = tree[parentNodeId].innerSplitPtr + (int)((tree[parentNodeId].outerSplitPtr - 1 - tree[parentNodeId].innerSplitPtr)/2.0);
			tInnerSplitPtr = t.innerSplitPtr;

			//Handle every left leaf node
			if(n == 1) {
				
				t.md = 0;
				t.idx = splitOfPoints[t.outerSplitPtr];
				tIdx = t.idx;

				for(int i=0; i<dim; i++) {

					t.vp[i] = x[(t.idx)*dim + i];
				}
				
				tree[id] = t;
				
				return;
			}
			//Handle every left non-leaf node
			else {

				t.idx = splitOfPoints[t.outerSplitPtr];
				tIdx = t.idx;

				for(int i=0; i<dim; i++) {

					t.vp[i] = x[(t.idx)*dim + i];
				}
			}	
		}
		//A right node (with nodeId an even number)
		else {

			t.innerSplitPtr = tree[parentNodeId].innerSplitPtr + (int)((tree[parentNodeId].outerSplitPtr - 1 - tree[parentNodeId].innerSplitPtr)/2.0) + 1;
			t.outerSplitPtr = tree[parentNodeId].outerSplitPtr - 1;
			tInnerSplitPtr = t.innerSplitPtr;

			//Handle every right leaf node
			if(n == 1) {
				
				t.md = 0;
				t.idx = splitOfPoints[t.outerSplitPtr];
				tIdx = t.idx;

				for(int i=0; i<dim; i++) {

					t.vp[i] = x[(t.idx)*dim + i];
				}
				
				tree[id] = t;
				
				return;
			}
			//Handle every right non-leaf node
			else {

				t.idx = splitOfPoints[t.outerSplitPtr];
				tIdx = t.idx;

				for(int i=0; i<dim; i++) {

					t.vp[i] = x[(t.idx)*dim + i];
				}
			}
		}
	}
	
	//Make worker threads wait for master thread to finish the common work
	__syncthreads();

	//Find the euclidean distance of this point to all the other points of this sub-tree
	//Each thread will take up dive points, except for the last thread that will take up dive+mod points
	int dive = (int)( (n-1)/numThreads );
	int mod = (n-1)%numThreads;
	int thrOffset = tInnerSplitPtr + threadId*dive;

	if(threadId != (numThreads - 1)) {
		
		for(int i=thrOffset; i<thrOffset+dive; i++) {

			double currD = 0;
			for(int j=0; j<dim; j++) {

				currD = currD + (double) pow((x[splitOfPoints[i]*dim + j] - x[(tIdx)*dim + j]),2);
			}
			d[i] = (double) sqrt(currD);
		}
	}
	else {
		
		for(int i=thrOffset; i<thrOffset+dive+mod; i++) {

			double currD2 = 0;
			for(int j=0; j<dim; j++) {

				currD2 = currD2 + (double) pow((x[splitOfPoints[i]*dim + j] - x[(tIdx)*dim + j]),2);
			}
			d[i] = (double) sqrt(currD2);
		}

	}

	//Syncrhonize the distance calculation work
	__syncthreads();

	//Calculate the median distance and split to inner/outer (only from the master thread)
	if(threadId == 0) {

		double *tempD = (double *)malloc((n-1)*sizeof(double));
		for(int i=0; i<n-1; i++){

			tempD[i] = d[tInnerSplitPtr + i];
		}

		//Calculate the median distance
		double medianDistance = quickselect(tempD, 0, n-2, k-1-(n%2));
		t.md = medianDistance;

		//Keep a new array for the inner sub-tree indexes
		int *innerIdxs = (int *)malloc((k-(n%2)) * sizeof(int));
		int counterIn = 0;

		//Keep a new array for the outer sub-tree indexes
		int *outerIdxs = (int *)malloc((k-1) * sizeof(int));
		int counterOut = 0;

		//Build the two new sub-arrays of point indexes
		for(int i=0; i<n-1; i++){

			if(d[tInnerSplitPtr + i] <= medianDistance) {

				innerIdxs[counterIn] = splitOfPoints[tInnerSplitPtr + i];
				counterIn++;
			}
			else {

				outerIdxs[counterOut] = splitOfPoints[tInnerSplitPtr + i];
				counterOut++;
			}
		}

		//Re-order the split of points to group per inner and outer
		for(int i=tInnerSplitPtr; i<tInnerSplitPtr + counterIn; i++){

			splitOfPoints[i] = innerIdxs[i - tInnerSplitPtr];
		}
		for(int i=tInnerSplitPtr + counterIn; i<t.outerSplitPtr; i++){

			splitOfPoints[i] = outerIdxs[i - tInnerSplitPtr - counterIn];
		}

		//Assign the struct vptree t to the corresponding node of the tree
		tree[id] = t;

		//Free some memory from this node and from the parent node
		free(tempD);
		free(innerIdxs);
		free(outerIdxs);
	}

	//Make worker threads wait for master thread to finish the common work
	__syncthreads();

	return;
}

/**
* Main CUDA kernel that builds the vantage point tree
*
* @param n: Number of data points (rows of X)
* @param dim: Number of dimensions (columns of X)
* @param X: Input data points
* @param splitOfPoints: An array of length n that holds the indexes of the initial points to positions related to their corresponding node
* @param d: An array of length n that holds the distances for the initial points to positions related to their corresponding node
* @param idxs: Array of indexes to the original positions of the set of points
* @param tree: Array of struct vptree nodes, each containing the needed information
* @param depth: Current depth of the tree to be built
* @param maxDepth: Maximum depth of the tree
* @param offset: Offset to add when calculating the node's index
*
**/

__global__ void buildvp(double *X, struct vptree *tree, int* splitOfPoints, double* d, int n, int dim, int depth, int maxDepth, int offset)
{
	//Each CUDA thread takes care of a node of the VP tree of this depth
	int nodeId = (pow(2,depth) - 1) + blockIdx.x + offset;
	if(nodeId > pow(2,maxDepth+1) - 2) return;

	//Strange bug with 2046 node appearing twice instead of 2048 ultra-mega-quick-fix
	if(nodeId == 2046){
		if(found == 0) {
			found = 1;
		}
		else {
			nodeId = 2048;
		}
	}

	//For the specific node, calculate all needed information and store them in the array
	vpt(n, dim, X, splitOfPoints, d, nodeId, depth, maxDepth, tree);	

	return;
}

/**
* Connects the VPs from treeArray to the initial points in the X dataset
*
* @param treeArray: The constructed array that represents the VP Tree
* @param dim: Number of dimensions (columns of X)
* @param X: Input data points
* @param len: Length of the X dataset
* @param treeLen: Length of the array that represents the VP Tree
*
**/

void makeConnectionsBetweenNodes(struct vptree *treeArray, double *X, int len, int treeLen, int dim) {

	for(int i=0; i<treeLen; i++){

		treeArray[i].vp = (double*)malloc(dim*sizeof(double));

		for(int j=0; j<dim; j++){

			if(treeArray[i].idx != -1){

				treeArray[i].vp[j] = X[treeArray[i].idx * dim + j];
			}
		}
	}
}
