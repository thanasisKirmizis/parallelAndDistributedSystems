#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

#define N 517
#define MAX_ERR 1e-6

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/
void ising( int *G, double *w, int k, int n){
  
    //Keep a copy of G to read from/write to
    int *G2 = malloc(n*n*sizeof(int));
    memcpy(G2, G, n*n*sizeof(int));

    int *temp;

    //Do k iterations
    for(int i=0; i<k; i++) {

        //For each point of the square lattice
        for(int m=0; m<n*n; m++) {

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

        //Swap the arrays for reading/writing
        temp = G2;
        G2 = G;
        G = temp;

    }

	//Use a memcpy before return or else the lattice would return empty
    memcpy(G, G2, n*n*sizeof(int));
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

	gettimeofday(&start, NULL);
    
	ising(G,w,11,N);

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

    //Deallocate host memory
    free(G); 
    free(G1);

    return 0;
}
