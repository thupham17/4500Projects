#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

//generate a feasible allocation
int getFeasible(int *pn, double *plb, double *pub, double *pmu, double *pcovar double *px)
{
    int sum = 0;
    x = &plb;
    upper = &pub;
    lower = &plb;

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
              printf("Bounds are not feasible.");
              break
          }
            else {
                x[j] = upper[j];
                delta = upper[j] - lower[j];
                sumx += upper[j] - lower[j];
            }
        }
    }
}

\\ Find F
int getF(int *pn, double x, double *pmu, double *pcov, double *lambda) {
    int cov = 0;
    int mean = 0;
    for (int i = 0; i < &pn; i++) {
        for (int i = 0; i < &pn; i++) {
        }
      }
}
int findS(double *px, double *ps, double *py, double *pmu, double *pcov) {
    int G0 =
}
//creates a new file containing the sorted array arr
void newfile(char *s1, char *s2, int size, float arr[])
{
}
