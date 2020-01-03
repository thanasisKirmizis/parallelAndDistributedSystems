#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <cuda.h>

#define N 517
#define MAX_ERR 1e-6
#define BLOCK_DIM 11	//must be integer divisor of N

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/
__global__ void ising(int *G, int *G2, double *w, int n){

	//Each CUDA block takes care of BLOCK_DIM*BLOCK_DIM points of the square lattice
	int m_x = blockIdx.x * blockDim.x + threadIdx.x;
	int m_y = blockIdx.y * blockDim.y + threadIdx.y;

    for(int i=0; i<BLOCK_DIM; i++) {

        for(int j=0; j<BLOCK_DIM; j++) {

			//Index of lattice point
            int m = (i+m_y*BLOCK_DIM)*n + (j+m_x*BLOCK_DIM);

            //Calculate the weighted influence of its neighbors
            double infl = 0;

            for(int a=0; a<5; a++) {

                for(int b=0; b<5; b++) {

                    //$w{0,0} is undefined
                    if(a == 2 && b == 2) {

                        continue;
                    }
                    
                    int x = (n+m+b-2)%n;
                    int y = (n+m/n+a-2)%n;
                    int idx =  x + y*n;

                    infl = infl + w[a*5 + b]*G2[idx];
                } 
            }

            if(fabs(infl) < MAX_ERR) {

                G[m] = G2[m];
            }
            else if(infl < 0) {
                
                G[m] = -1;
            }
            else if(infl > 0) {

                G[m] = 1;
            }   
            
        }

    }

}


int main() {

	struct timeval start, end;

    //Allocate host memory
    int *G = (int*) malloc(N*N*sizeof(int));

    double w[25] = {0.004,  0.016,  0.026,  0.016,   0.004,
                 0.016,  0.071,  0.117,  0.071,   0.016,
                 0.026,  0.117,  0,      0.117,   0.026,
                 0.016,  0.071,  0.117,  0.071,   0.016,
                 0.004,  0.016,  0.026,  0.016,   0.004};

    //Initialize the lattice
    FILE *fp = fopen("conf-init.bin", "rb");
    fread(G, sizeof(int), N * N, fp);
    fclose(fp);

    //Allocate device memory
    int *d_G;
    cudaMalloc((void**)&d_G, N*N*sizeof(int));
    int *d_G2;
    cudaMalloc((void**)&d_G2, N*N*sizeof(int));
    double *d_w;
	cudaMalloc((void**)&d_w, 25*sizeof(double));

    //Transfer data from host to device memory
    cudaMemcpy(d_G, G, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G2, G, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, 25*sizeof(double), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock( 1, 1 );
    dim3 numBlocks( N/(BLOCK_DIM*threadsPerBlock.x), N/(BLOCK_DIM*threadsPerBlock.y) );
    
	gettimeofday(&start, NULL);

    for(int i=0; i<11; i++) {

        ising<<<numBlocks, threadsPerBlock>>>(d_G,d_G2,d_w,N);
        cudaMemcpy(d_G2, d_G, N*N*sizeof(int), cudaMemcpyDeviceToDevice);
    }

    //Transfer data back to host memory
    cudaMemcpy(G, d_G, N*N*sizeof(int), cudaMemcpyDeviceToHost);

	gettimeofday(&end, NULL);

	double time_spent = (end.tv_sec - start.tv_sec) + 
						(end.tv_usec - start.tv_usec) / 1e+6;

	printf("Time spent: %f\n", time_spent);

	//Verify results
	int *G1 = (int*) malloc(N*N*sizeof(int));

	fp = fopen("conf-11.bin", "rb");
	fread(G1, sizeof(int), N * N, fp);
	fclose(fp);

	int noobcnt = 0;
	for(int i=0; i<N*N; i++) {

		if(G[i] != G1[i]) {

		    noobcnt++;
		}
	}
	printf("\n\nWrong Elements: %d\n\n", noobcnt);

    //Deallocate device memory
    cudaFree(d_G);
    cudaFree(d_w);

    //Deallocate host memory
    free(G); 
    free(G1);

    return 0;
}
