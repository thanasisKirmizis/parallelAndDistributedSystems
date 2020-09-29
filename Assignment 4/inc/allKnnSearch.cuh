/**

    Author:  	Athanasios Kirmizis
    Dept.:   	EE AUTH
    AEM :    	8835
    Course:  	Parallel And Distributed Systems
	Assignment: #4
	Season:  	2019 - 2020
    E-mail : 	athakirm@ece.auth.gr
    Prof:    	Nikolaos Pitsianis | Dimitrios Floros

    Functions to Perform all-Knn Search on a Vantage Point Tree Using CUDA

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
#define SWAP2(x, y) { int temp = x; x = y; y = temp; }

/**
* Puts an element in the array neighborsIdxs and sorts the array based on the sorted order of neighborsDists
*
* @param neighborsIdxs: Array of indexes to the initial dataset of already searched points
* @param neighborsDists: Array of distances of already searched points to the query point
* @param pointIdx: Index to the initial dataset of the query element that kNN search is being applied to
* @param newIdx: Index to the initial dataset of the element to be inserted in the array
* @param newDist: Distance to the query point of the element to be inserted in the array
* @param k: The kNN parameter
*
**/

__device__ void putAndAutoSort(int *neighborsIdxs, double *neighborsDists, int pointIdx, int newIdx, double newDist, int k) {

	//Offset so that each CUDA thread will handle its own part of the array
	int arrayOffset = pointIdx*k;

	//Iterate through array to find the correct place of the new point
	for(int i=0; i<k; i++) {
		
		//This means that array is empty or in ascending order
		if(neighborsDists[i] == -1.0) {
			
			neighborsDists[i] = newDist;
			neighborsIdxs[arrayOffset + i] = newIdx;
			break;
		}
		//This means that array is not in ascending order
		else if(newDist < neighborsDists[i]) {
			
			//Correct place found, so shift remaining elements to the right or stop if -1 is found
			for(int j=i+1; j<k; j++) {
				
				SWAP(neighborsDists[i],neighborsDists[j]);
				SWAP2(neighborsIdxs[arrayOffset + i],neighborsIdxs[arrayOffset + j]);
				
				if(neighborsDists[i] == -1.0) break;
			}
			
			//Replace the last element with the new one and return
			neighborsDists[i] = newDist;
			neighborsIdxs[arrayOffset + i] = newIdx;
			break;
		}
	}
	
	return;
}

/**
* Helper function to calculate distance between two [dim]-dimensional points
*
* @param p1: Point 1
* @param p2: Point 2
* @param dim: Dimension of points
*
**/

__device__ double distFun(double *p1, double *p2, int dim) {
	
	double d = 0;
	for(int j=0; j<dim; j++) {

		d = d + (double) pow((p1[j] - p2[j]),2);
	}
	d = (double) sqrt(d);

	return d;
}

/**
* Helper function to reallocate memory inside GPU kernels
*
* @param oldSize: Size of array before reallocation
* @param newSize: Size of array after reallocation
* @param old: Pointer to old array
*
**/

__device__ int* myRealloc(int oldSize, int newSize, int* old)
{
	int* newT = (int*) malloc(newSize*sizeof(int));
	
    for(int i=0; i<oldSize; i++)
    {
        newT[i] = old[i];
    }

    free(old);
    return newT;
}

/**
* Finds the kNNs of all points of a dataset X using a CUDA thread for each point and by searching through a VP Tree
*
* @param dim: Dimension of points of X
* @param k: The kNN parameter
* @param X: The dataset of points
* @param treeArray: The constructed array that represents the VP Tree
* @param knnNeighbors: The array to be filled with the kNNs of each point
* @param offset: Offset to add when calculating the node's index
* @param maxDepth: Maximum depth of the tree
*
**/

__global__ void findKnnNeighbors(int dim, int k, double *X, struct vptree *treeArray, int *knnNeighbors, int offset, int maxDepth) {
	
	//Each CUDA thread takes care of a point of the initial dataset X
	int nodeId = blockIdx.x * blockDim.x + threadIdx.x + offset;
	if(nodeId > pow(2,maxDepth)-2) return;
	
	int pointIdx = treeArray[nodeId].idx;
	
	//Ignore the Null Nodes of the tree as they don't correspond to any point in X
	if(pointIdx == -1) return;

	//Initialize ids of neighbors to -1
	for(int i=0; i<k; i++) {
		
		knnNeighbors[pointIdx*k + i] = -1;
	}

	//Define the query point for this CUDA thread
	double *query = (double*)malloc(dim*sizeof(double));
	for(int j=0; j<dim; j++){

		query[j] = X[pointIdx * dim + j];
	}

	//Initialize an array for distances to the k nearest points found until now
	double *neighborsDists = (double*)malloc(k*sizeof(double));
	for(int i=0; i<k; i++) {
		
		neighborsDists[i] = -1.0;
	}

	//Initialize some variables for the searching of the tree
	int numOfNodesToSearch = 1;
	int currNodesPtr = 0;
	int maxNodesPtr = 0;
	int *nodesToSearch = (int*)malloc(sizeof(int));
	double dist;
	double furthest_d = DBL_MAX;
	int currNodeId;
	int furthestPointIdx;
	double* furthestPointVp = (double*)malloc(dim*sizeof(double));
	int nFoundUntilNow = 0;

	//Start the search from the root node
	nodesToSearch[0] = 0;
	
	//While not all kNNs have been found
	while(numOfNodesToSearch > 0) {

		//Get the next node to search
		currNodeId = nodesToSearch[currNodesPtr];
		currNodesPtr = currNodesPtr + 1;
		numOfNodesToSearch = numOfNodesToSearch - 1;
		
		//If node is Null, ignore it
		if(treeArray[currNodeId].idx == -1) continue;
		
		//Calculate the distance of the query to this node
		dist = distFun(treeArray[currNodeId].vp, query, dim);
		
		//If it is less than current furthest distance, put node in current kNNs and update furthest distance to current found furthest
		if(dist < furthest_d) {

			putAndAutoSort(knnNeighbors, neighborsDists, pointIdx, treeArray[currNodeId].idx, dist, k);
			nFoundUntilNow = nFoundUntilNow + 1;
			
			if(nFoundUntilNow >= k) {

				furthestPointIdx = knnNeighbors[pointIdx*k + k-1];
				for(int j=0; j<dim; j++) {
				
					furthestPointVp[j] = X[furthestPointIdx * dim + j];
				}
				furthest_d = distFun(furthestPointVp, query, dim); 
			}
		}

		//If node is Leaf, go to next node
		if(treeArray[currNodeId].md == 0) continue;
		
		//If distance of query to point of node is less than node.md + furthest distance, then add left child of this node to be searched
		if(dist <= treeArray[currNodeId].md + furthest_d) {
			
			numOfNodesToSearch = numOfNodesToSearch + 1;
			maxNodesPtr = maxNodesPtr + 1;
			nodesToSearch = myRealloc(maxNodesPtr, maxNodesPtr+1, nodesToSearch);
			nodesToSearch[maxNodesPtr] = 2*currNodeId + 1;
		}
		
		//If distance of query to point of node is more or equal than node.md - furthest distance, then add right child of this node to be searched
		if(dist >= treeArray[currNodeId].md - furthest_d) {
			
			numOfNodesToSearch = numOfNodesToSearch + 1;
			maxNodesPtr = maxNodesPtr + 1;
			nodesToSearch = myRealloc(maxNodesPtr, maxNodesPtr+1, nodesToSearch);
			nodesToSearch[maxNodesPtr] = 2*currNodeId + 2;
		}
	}

	//Free and return
	free(query);
	free(neighborsDists);
	free(nodesToSearch);
	free(furthestPointVp);

	return; 
}
