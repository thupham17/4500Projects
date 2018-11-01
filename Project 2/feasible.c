#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

double *arr;

//generate a feasible allocation
double * getFeasible(int* n, double * lb, double * ub, double * mu, double * covar) {
    double sum = 0;
    int i, j;

    double * x = malloc(*n * sizeof(double));

    for (int i = 0; i < *n; i++) {
        x[i] = lb[i];
        sum += x[i];
    }

    if (sum > 1){
        printf("Bounds are not feasible.");
        exit(0);
    }

    else if (sum == 1){
        return x;
    }

    else {
        int delta = 0;
        for (int j = 0; j < *n; j++) {
            if (sum + (ub[j] - lb[j]) >= 1.0) {
              x[j] = 1.0 - sum + lb[j];
              printf("Done.");
              break;
          }
            else {
                x[j] = ub[j];
                delta = ub[j] - lb[j];
                sum += ub[j] - lb[j];
            }
        }
        return x;
    }
}

// get F at vector x
double getFx(int n, double *mu, double *cov, double lambda, double *x) {
    double scov = 0;
    double smean = 0;
    for (int i = 0; i < n; i++) {
        smean = smean + (mu[i])*(x[i]);
        for (int j = 0; j <= i; j++) {
            if (j == i)
            {
            	scov += (cov[i* (n) + j]) * pow((x[i]), 2);
            }
            else
            {
            	scov += 2*(cov[i* n + j])*(x[i])*(x[j]);
            }
        }
    }

    double F = scov*lambda - smean;
    return F;
}

// get gradient of F at (x)
double * getdFx(int n, double * mu, double * covariance, double lambda, double * x)
{
	double * grad = malloc(n * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++) {
			grad[i] += 2*lambda*covariance[i*n + j]*x[j];
		}
		grad[i] -= mu[i];
	}
	return grad;
}

// utility function for getting Y
int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (arr[ia] < arr[ib]) - (arr[ia] > arr[ib]);
}
//step 1 find a vector y(k)
double * getdX(int n, double *mu, double *cov, double lambda, double * lb, double * ub, double *x)
{
    // get gradF(x)
    double * grad = getdFx(n, mu, cov, 10, x);

    int ind[n];
    double g, gbest, ysum;
    int feascount = 0;
    double * y = malloc(n*sizeof(double));
    double * ybest = malloc(n*sizeof(double));

    // get original indices of sorted elements in grad
    for(int i = 0; i < n; ++i)
        ind[i] = i;

    arr = grad;
    qsort(ind, n, sizeof(*ind), cmp);

    for(int m = 0; m < n; m++) {
        y[m] = 0;
        g = 0;
        ysum = 0;
        for (int j = 0; j < n; j++){
          if (j < m) {
              y[j] = lb[ind[j]] - x[ind[j]];
              ysum += y[j];
              g += y[j]*grad[ind[j]];
          }
          else if (j> m) {
              y[j] = ub[ind[j]] - x[ind[j]];
              ysum += y[j];
              g += y[j]*grad[ind[j]];
          }
        }
        y[m] = 0 - ysum;
        g += y[m]*grad[ind[m]];

        if ((lb[ind[m]] - x[ind[m]] <= y[m]) && (y[m] <= ub[ind[m]] - x[ind[m]])) {
            // set ybest
            if (feascount == 0) {
              gbest = g;
              //convert to old index
              for(int k = 0; k < n; k++) {
                  ybest[ind[k]] = y[k];
              }
            }
            // new best candidate y
            else if (g < gbest) {
              for(int k = 0; k < n; k++) {
                  ybest[ind[k]] = y[k];
              }
              gbest = g;
            }
            feascount++;

        }
    }

    if (feascount == 0)
    {
      printf("Program terminated. Optimal value found.\n");
    }
    free(y);
    return ybest;
}

double findS(int n, double *x, double *y, double *mu, double *cov, double lambda) {
    double xycov = 0;
    double meany = 0;
    double denom = 0;
    double s = 0;

    // solve for s using first order condition
    for (int i = 0; i < n; i++) {
        meany = meany + mu[i]*y[i];
        denom = denom + (cov[i*n + i]*pow(y[i], 2));
        xycov = xycov + (cov[i*n +i])*x[i]*y[i];
        for (int j = 0; j < i; j++) {
                xycov = xycov + cov[i*n + j]*(x[i]*y[j]+x[j]*y[i]);
                denom = denom + 2*cov[i*n + j]*y[i]*y[j];
            }
     }

    s = (meany - 2*lambda*xycov)/(2*lambda*denom);

    // check boundary conditions
    if (s < 0)
        s = 0;
    else if (s > 1)
        s = 1;

    return s;
}
