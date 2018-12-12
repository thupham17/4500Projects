#include <Windows.h>
#include <stdio.h>
#include <stdlib.h> 


#include "baggie.h"

// a made up, slow, numerically intense computation:
void comp(baggie *pbag)
//void comp(double v1, double v2, double v3, int name, double *presult)
{
	double v1, v2, v3;
	int name;
	double *presult;
	int i, j, k, M, iteration, itsdone = 0;
	double totaliterations;
	double value;

	v1 = pbag->v1;
	v2 = pbag->v2;
	v3 = pbag->v3;
	name = pbag->bagname;
	presult = &pbag->result;

	int mycount = 0;

	M = 10000;
	value = 0;
	iteration = 0;
	totaliterations = 0;
	for(i = 0; i < M; i++){
		for(j = 0; j < M; j++){
			for(k = 0; k < M; k++){
				value += i*v1 + j*j*v2 + k*k*k*v3;
				++iteration;
				if(iteration == 90000000){

					WaitForSingleObject(pbag->consolemutex, INFINITE);
					printf("%d : value %g, total iterations %g mycount %d\n", name, value,
						totaliterations, mycount); 
				    //if (mycount < 50 )
						ReleaseMutex(pbag->consolemutex);
					mycount += 1;

					WaitForSingleObject(pbag->resultmutex, INFINITE);
					*presult = value;
					ReleaseMutex(pbag->resultmutex);
					iteration = 0;
				}
				++totaliterations;
				if(totaliterations == 5000000000000){
					itsdone = 1;
				}

				if(itsdone) goto DONE;
				
			}
		}
	}

DONE:
	*presult = value;
	printf("alue = %f. done\n",value);
}
