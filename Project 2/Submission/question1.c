#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "methods.h"

int readit(char *nameoffile, int *addressofn, double **, double **, double **, double **);

int main(int argc, char **argv)
{
    int retcode = 0;
    int n;
    double *lb, *ub, *covariance, *mu, lambda;
    double Fnew;

    if (argc != 2){
    	  printf("usage: qp1 filename\n");  retcode = 1;
    	  goto BACK;
      }

    retcode = readit(argv[1], &n, &lb, &ub, &mu, &covariance);
    lambda = 10;

    double s, F, ynorm;
    ynorm = 0;
    // get feasible x
    double *x = getFeasible(&n, lb, ub, mu, covariance);
    printf("x0: ");
    for (int i = 0; i < n; i++) {
        printf("%g ", x[i]);
    }
    printf("\n");

    Fnew = getFx(n, mu, covariance, 10, x);
    printf("F = %g\n",Fnew);

    int iter = 0;
    printf("\n");
    while (iter == 0 || iter < 100000) {
        printf("Iteration: %d\n",iter);
        printf("diff = %g",Fnew - F);
        double * y = getdX(n, mu, covariance, lambda, lb, ub, x);
        ynorm = 0;
        printf("y: ");
        for (int i = 0; i < n; i++) {
            ynorm += pow(y[i],2);
            printf("%g ", y[i]);
        }
        printf("\n");

        s =  findS(n, x, y, mu, covariance, 10);
        printf("s = %g\n",s);

        printf("x: ");
        for (int i = 0; i < n; i++) {
            x[i] = x[i] +s*y[i];
            printf("%g ", x[i]);
        }
        printf("\n");

        F = Fnew;
        Fnew = getFx(n, mu, covariance, 10, x);
        printf("F = %g\n",Fnew);
        iter++;
    }


    for (int i = 0; i < n; i++) {
      printf("%g\n",x[i]);
    }
    F = getFx(n, mu, covariance, 10, x);
    printf("F = %g",F);
    BACK:
    return retcode;
}

int readit(char *filename, int *address_of_n, double **plb, double **pub,
		double **pmu, double **pcovariance)
{
	int readcode = 0, fscancode;
	FILE *datafile = NULL;
	char buffer[100];
	int n, i, j;
	double *lb = NULL, *ub = NULL, *mu = NULL, *covariance = NULL;

	datafile = fopen(filename, "r");
	if (!datafile){
		printf("cannot open file %s\n", filename);
		readcode = 2;  goto BACK;
	}

	printf("reading data file %s\n", filename);

	fscanf(datafile, "%s", buffer);
	fscancode = fscanf(datafile, "%s", buffer);
	if (fscancode == EOF){
		printf("problem: premature file end at ...\n");
		readcode = 4; goto BACK;
	}

	n = *address_of_n = atoi(buffer);

	printf("n = %d\n", n);

	lb = (double *)calloc(n, sizeof(double));
	*plb = lb;
	ub = (double *)calloc(n, sizeof(double));
	*pub = ub;
	mu = (double *)calloc(n, sizeof(double));
	*pmu = mu;
	covariance = (double *)calloc(n*n, sizeof(double));
	*pcovariance = covariance;

	if (!lb || !ub || !mu || !covariance){
		printf("not enough memory for lb ub mu covariance\n"); readcode = 3; goto BACK;
	}

	fscanf(datafile, "%s", buffer);

	for (j = 0; j < n; j++){
		fscanf(datafile, "%s", buffer);
		fscanf(datafile, "%s", buffer);
		lb[j] = atof(buffer);
		fscanf(datafile, "%s", buffer);
		ub[j] = atof(buffer);
		fscanf(datafile, "%s", buffer);
		mu[j] = atof(buffer);
		printf("j = %d lb = %g ub = %g mu = %g\n", j, lb[j], ub[j], mu[j]);
	}

	fscanf(datafile, "%s", buffer);

  fscanf(datafile, "%s", buffer);

	fscanf(datafile, "%s", buffer);

  /* reading 'covariance'*/
  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++){
      fscanf(datafile, "%s", buffer);
      covariance[i*n + j] = atof(buffer);
    }
  }

	fscanf(datafile, "%s", buffer);
	if (strcmp(buffer, "END") != 0){
		printf("possible error in data file: 'END' missing\n");
	}


	fclose(datafile);

BACK:
	return readcode;
}
