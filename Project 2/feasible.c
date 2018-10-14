#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

//generate a feasible allocation
void getFeasible(int* n, double * lb, double * ub, double * mu, double * covar, double * x) {
    printf("getFeasible\n");
    double sum = 0;
    int i, j;

    for (int i = 0; i < *n; i++) {
        x[i] = lb[i];
        sum += x[i];
    }

    if (sum > 1){
        printf("Bounds are not feasible.");
        exit(0);
    }

    else if (sum == 1){
        return;
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
        return;
    }
}

// Find F
int getF(int *n, double *x, double *mu, double *covar, double *lambda) {
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
