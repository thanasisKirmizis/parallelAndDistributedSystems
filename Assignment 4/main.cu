/**

    Author:  	Athanasios Kirmizis
    Dept.:   	EE AUTH
    AEM :    	8835
    Course:  	Parallel And Distributed Systems
	Assignment: #4
	Season:  	2019 - 2020
    E-mail : 	athakirm@ece.auth.gr
    Prof:    	Nikolaos Pitsianis | Dimitrios Floros

    The Main Function to Run from

**/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include "inc/vptree.h"
#include "inc/constructVpTree.cuh"
#include "inc/allKnnSearch.cuh"

#define LEN 10000
#define DIMEN 2
#define K 5
#define NUMTHREADS 32
#define NUMBLOCKS 64

int main() {

	//Ask the user whether or not to display detailed results
	int opt = 1;
  	printf("Enter 1 to show only timings of kernels and correctness of results or 2 to show detailed results: ");
	scanf("%d",&opt);
	if(opt != 1 && opt != 2) opt = 1;

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int mb = prop.totalGlobalMem/pow(2,20);
	printf("\nInfo: Graphic Card's Global Memory: %d MB\n\n", mb);

	srand(time(NULL));
	struct timeval start, end, start2, end2;

	//------------------------------ Firstly construct the tree --------------------------//

	//Calculate the depth of the tree to be constructed
	int treeDepth = (int)log2(LEN);
	if(opt == 2) printf("Tree Depth: %d\n",treeDepth);
	int fullTreeLen = pow(2,treeDepth+1) - 1;

	//Allocate host memory
	double *X = (double *)malloc(LEN*DIMEN*sizeof(double));
	struct vptree *treeArray = (struct vptree*)malloc(fullTreeLen*sizeof(struct vptree));
	int* splitOfPoints = (int *)malloc(LEN*sizeof(int));

	//Initialize the array of n-dimensional points
	for (int i=0; i<LEN; i++) {

		splitOfPoints[i] = i;

		for(int j=0; j<DIMEN; j++) {

			X[i*DIMEN + j] = rand();
			X[i*DIMEN + j] = X[i*DIMEN + j]/RAND_MAX;
		}
	}

    //Allocate device memory
	double *d_X;
	double *d_d;
	struct vptree *d_treeArray;
	int *d_splitOfPoints;
	cudaMalloc((void**)&d_X, LEN*DIMEN*sizeof(double));
	cudaMalloc((void**)&d_d, LEN*sizeof(double));
	cudaMalloc((void**)&d_splitOfPoints, LEN*sizeof(int));
	cudaMalloc((void**)&d_treeArray, fullTreeLen*sizeof(struct vptree));

    //Transfer data from host to device memory
	cudaMemcpy(d_X, X, LEN*DIMEN*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitOfPoints, splitOfPoints, LEN*sizeof(int), cudaMemcpyHostToDevice);
    
	gettimeofday(&start, NULL);

	//Construct the VP tree, one level at a time
    for(int i=0; i<=treeDepth; i++) {

		if(opt == 2) printf("\nBuilding Level %d...\n",i);
		if(opt == 2) printf("--------------------------\n");

		//If more than 2^12 blocks are about to be created, cap them at 2^12 at a time
		if(i <= 12) {

			buildvp<<<pow(2,i), NUMTHREADS>>>(d_X, d_treeArray, d_splitOfPoints, d_d, LEN, DIMEN, i, treeDepth, 0);
			cudaDeviceSynchronize();
		}
		else {

			for(int j=0; j<pow(2,i-12); j++) {

				buildvp<<<pow(2,12), NUMTHREADS>>>(d_X, d_treeArray, d_splitOfPoints, d_d, LEN, DIMEN, i, treeDepth, pow(2,12)*j);
				cudaDeviceSynchronize();
			}
		}
    }

    //Transfer data back to host memory
    cudaMemcpy(treeArray, d_treeArray, fullTreeLen*sizeof(struct vptree), cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);

	double time_spent = (end.tv_sec - start.tv_sec) + 
						(end.tv_usec - start.tv_usec) / 1e+6;

	printf("Tree Build Complete!\n\nTime spent inside kernel: %f\n\n", time_spent);

	//Connect the VPs from treeArray to the initial points in the X dataset
	makeConnectionsBetweenNodes(treeArray, X, LEN, fullTreeLen, DIMEN);

	//Custom VP Tree construction mini-tester
	long sumNums = 0;
	long sumIdxs = 0;
	for(int i=0; i<fullTreeLen; i++) {
		
		if(treeArray[i].idx != -1) sumIdxs = sumIdxs + treeArray[i].idx;
	}
	for(int i=0; i<LEN; i++) {

		sumNums = sumNums + i;
	}
	printf("Asserting if all idxs are found in the VP Tree...\n");
	if(sumIdxs == sumNums) printf("CONSTRUCTION TEST PASSED!\n\n");
	else printf("CONSTRUCTION TEST FAILED!\n\n");

	//-------------------------------------------------------------------------------------//

	//------------------------------ Then search kNNs through the tree --------------------------//
	
	//Allocate host and device memory
	int *allKnnNeighbors = (int*)malloc(LEN*K*sizeof(int));
	int *d_allKnnNeighbors;
	cudaMalloc((void**)&d_allKnnNeighbors, LEN*K*sizeof(int));

	gettimeofday(&start2, NULL);

	//Call the kernel to search through the tree for the kNNs of each of the points
	//If more than (NUMBLOCKS*NUMTHREADS) threads are about to be created, cap them at (NUMBLOCKS*NUMTHREADS) at a time
	int steps = (int)(fullTreeLen+1)/(NUMBLOCKS*NUMTHREADS);
	if(steps > 0) {

		for(int j=0; j<steps; j++) {

			findKnnNeighbors<<<2*NUMBLOCKS, NUMTHREADS/2>>>(DIMEN, K, d_X, d_treeArray, d_allKnnNeighbors, (NUMBLOCKS*NUMTHREADS)*j, treeDepth+1);
		}
		cudaDeviceSynchronize();
	}
	else {

		findKnnNeighbors<<<2*NUMBLOCKS, NUMTHREADS/2>>>(DIMEN, K, d_X, d_treeArray, d_allKnnNeighbors, 0, treeDepth+1);
		cudaDeviceSynchronize();
	}

	gettimeofday(&end2, NULL);

	double time_spent2 = (end2.tv_sec - start2.tv_sec) + 
						(end2.tv_usec - start2.tv_usec) / 1e+6;

	//Transfer data back to host memory
    cudaMemcpy(allKnnNeighbors, d_allKnnNeighbors, LEN*K*sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	//Print results
	if(opt == 2) {

		for(int i=0; i<LEN; i++) {

			printf("K = %d nearest neighbors to X[%d] = (%f,%f) are:\n", K, i, X[i*DIMEN], X[i*DIMEN + 1]);
			for(int j=0; j<K; j++){
				if(allKnnNeighbors[i*K + j] != -1) {

					printf("(%f,%f)\n", X[allKnnNeighbors[i*K + j]*DIMEN], X[allKnnNeighbors[i*K + j]*DIMEN + 1]);
				}
			}
			printf("-----------------------\n\n");
		}
	}

	printf("All-kNN Search Complete!\n\nTime spent inside kernel: %f\n\n", time_spent2);

	//Custom All-kNN Search mini-tester
	int flag = 1;
	for(int j=1; j<K; j++){
		if(allKnnNeighbors[(LEN-1)*K + j] != -1) {

			if(X[allKnnNeighbors[(LEN-1)*K + j]*DIMEN] == X[allKnnNeighbors[(LEN-1)*K + j-1]*DIMEN]) {

				flag = 0;
			}
		}
	}
	printf("Checking if memory error occured...\n");
	if(flag == 1) printf("ALL-KNN SEARCH TEST PASSED!\n\n");
	else printf("ALL-KNN SEARCH TEST FAILED!\n\n");
	
	//-------------------------------------------------------------------------------------//
	
    //Deallocate device memory
	cudaFree(d_allKnnNeighbors);
	cudaFree(d_X);
	cudaFree(d_treeArray);
	cudaFree(d_d);
	cudaFree(d_splitOfPoints);

	//Deallocate host memory
	free(splitOfPoints);
	free(treeArray);
    free(X);
	free(allKnnNeighbors);

    return 0;
}
