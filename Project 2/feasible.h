#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

double * getFeasible(int *, double *, double *, double *, double *);
double * getdFx(int, double *, double *, double , double *);
double getFx(int, double *, double *, double , double *);
int cmp(const void *, const void *);
double * getdX(int, double *, double *, double, double * , double * , double *);
double findS(int, double *, double *, double *, double *, double);
