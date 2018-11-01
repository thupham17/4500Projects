#include<iostream>
#include<fstream>
#include <vector>
#include <math.h>
#include <random>
#include<string>

using namespace std;

vector< vector<double> > getReturns(vector<vector<double> > , int, int);
vector< vector<double> >readPrices(char *, int *, int *);
vector< vector<double> > dot(vector<vector<double> >, vector<vector<double> > );
vector<vector<double> > add_subtract(vector<vector<double> >, vector<vector<double> >, int);
vector<vector<double> > multiply(vector<vector<double> >, vector<vector<double> >);
vector<vector<double> > transpose(vector<vector<double> >);
double getError(vector<vector<double> >, vector<vector<double> >);
