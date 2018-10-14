#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "insertionSort.h"

//generate a feasible allocation
double getFeasible(int *pn, double *plb, double *pub, double *pmu, double *pcovar, double *px) {
    int sum = 0;
    double x = *plb;
    double upper = *pub;
    double lower = *plb;

    for (int i = 0; i < n; i++)
    sum += x[i];

    if (sum > 1){
        printf("Bounds are not feasible.");
        exit(0);
    }

    else if (sum = 1){
        *px = &x;
        break
    }

    else {
        for (int j = 0; j < ; j++) {
            if (sum + (upper[j] - lower[j]) >= 1.0) {
              x[j] = 1.0 - sum + lower[j];
              printf("Done.");
              break
          }
            else {
                x[j] = upper[j];
                delta = upper[j] - lower[j];
                sumx += upper[j] - lower[j];
            }
        }
        *px = &x;
    }
}

\\ Find F
int getF(int *pn, double *px, double *pmu, double *pcov, double *lambda) {
    double cov = 0;
    double mean = 0;
    for (int i = 0; i < &pn; i++) {
        mean = mean + &pmu*x[i];
        for (int j = 0; j < = i; j++) {
            if (j == i)
                cov = cov + &pcov[i][j]^2*&px[i]^2;
            else
                cov = cov + 2*pcov[i][j]*&px[i]*&px[j];
        }
    }
    double F = cov + mean;
    return F;
}

int fxd1(double lambda, double *covariance, double *x, double *mu)
{

    int length;
    length = sizeof(x)/sizeof(x[0]);

	double *grad_vector;

	grad_vector = (double *)calloc(length, sizeof(double));

	int i = 0;
	int j = 0;

	double simple;
	double harder;
	double *pharder;
	pharder = &harder

	for (i = 0, i < length, i++)
	{
		simple = 2 * lambda * (*covariance[i][i] * (*x[i]) - (*mu[i]))
		harder = 0

		for (j = 0, j < length, j++)

		{
			if j!= i
			{

				harder += (*covariance[i][j]) * (*x[j]);
			}

			*harder = harder * 2 * lambda
		}
		grad_vector[i] = simple + harder
		}
		return grad_vector
	}
}

// step 2 find optimal step size
int findS(double *pn, double *px, double *ps, double *py, double *pmu, double *pcov) {
    x1 = arr[n];
    for (int i = 0; i < &pn; i++) {
      x1[i] = &px[i] + &py[i];
    }
    double G0 = getF(&px);
    double G1 = getF(&x1);

    double xycov = 0;
    double meany = 0;
    double denom = 0;
    for (int i = 0; i < &pn; i++) {
        meany = meany + &pmu*y[i];
        denom = pcov^2*y[i];
        for (int j = 0; j < = i; j++) {
            if (j == i)
                xycov = xycov + &pcov[i][j]^2*&px[i]*&py[i];
            else
                xycov = xycov + 2*(&pcov[i][j](*&px[i]*&py[j]+*&px[j]*&py[i]));
        }
     }
    s = (meany -2*lambda*xycov)/denom;

    //get G(s)
    x2 = arr[n];
    double Gs = 0;
    for (int i = 0; i < &pn; i++) {
      x2[i] = &px[i] + &py[i];
    }
    Gs = getF(x2);

    if Gs =
    return min(G0,Gs,G1);
}

int *arr;

int cmp(const void *a, const void *b){
    int ia = *(int *)a;
    int ib = *(int *)b;
    return (arr[ia] > arr[ib]) - (arr[ia] < arr[ib]);
}

int isfeasible(double *plb, double *pub, double *pmu, int *pcandidate, double *px)
{
	int counter = 0;
	double x = *px;
	double upper = *pub;
	double lower = *plb;
	int candidate = *pcandidate;

	for(int i = 0, i < (sizeof(candidate)/sizeof(candidate[0])), i++)
	{
		if (lower[j] - x[j] <= candidate[j] && upper[j] - x[j] >= candidate[j]
		{
			counter += 1;
		}

		if counter == (sizeof(candidate)/sizeof(candidate[0]))
		{
			return 1;
		}
		else
		{
			return 0;
		}

	}
}

double linprosolv(double *lower, double *upper, double *px)
{

  int double y = getdF(int *pn, double *px, double *pmu, double *pcov, double *lambda);
  int ind[*pn];
  int i;
  bool feas;
  for(i = 0; i < len; ++i)
    ind[i] = i;

  arr = a;
  qsort(ind, *pn, sizeof(*ind), cmp);

  double gsort = arr[*pn];
  for(i = 0; i < *pn; ++i)
    gsort[i] = ind[i];

	int *candidate;
	candidate = (int *)calloc((sizeof(x)/sizeof(x[0])), sizeof(int));
	//Candidate should be a list of lists instead!

	int m = 0;

	int counter = 0;

	for (int i = 0, i < (sizeof(x)/sizeof(x[0])), i++)
	{
		double *y;
		y = (double *)calloc((sizeof(x)/sizeof(x[0])), sizeof(double));


		for (int j = 0, j < (sizeof(x)/sizeof(x[0])), j++)
		{

			if (j < m)
			{
				y[j] = lower[j] - x[j];
			}

			else if (j > m)
			{
				y[j] = upper[j] - x[j];
			}

			else if (j == m)
			{
				y[j] = 0;
			}

		double sum;

		for (int k = 0, k < (sizeof(y)/sizeof(y[0])), k++)
		{

			sum = sum + y[i];

		}

		y[m] = 0 - sum

		if isfeasible(x, lower, upper, y) == 1
		{

      \\ change candidate y to original index
      for(j = 0; j < *pn ++j) {
          y[j][ind[i]] = y[j][i];
      }

			candidate[counter] = y;
			counter += 1;
		}
		m += 1;
		}
	}

//step 1 find a vector y(k)
double getdX(int *pn, double *px, double *pmu, double *pcov, double *lambda)
{
    int double y = getdF(int *pn, double *px, double *pmu, double *pcov, double *lambda);
    int ind[*pn];
    int i;
    bool feas;
    for(i = 0; i < len; ++i)
      ind[i] = i;

    arr = a;
    qsort(ind, *pn, sizeof(*ind), cmp);

    double gsort = arr[*pn];
    for(i = 0; i < *pn; ++i)
      gsort[i] = ind[i];

    for(i = 0; i < *pn ++i) {

      \\check y feasible
      if (*plb[i] - *px[i] <= y[i] && *pub[i] - *px[i] >= y[i])
          feas = true;

      \\ change candidate y to original index
      for(j = 0; j < *pn ++j) {
          y[j][ind[i]] = y[j][i];
      }
    }

    return y;

}
