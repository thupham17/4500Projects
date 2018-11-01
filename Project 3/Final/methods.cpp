#include<iostream>
#include<fstream>
#include <vector>
#include <math.h>
#include <random>
#include<string>

using namespace std;

vector< vector<double> > readPrices(char *filename, int *pn, int *pm)
{
    ifstream inFile;

    char label1[100];
    char label2[100];
    char temp[1000];
    int n, m;

  	inFile.open(filename);

    if (!inFile) {
        cerr << "Unable to open file\n" << endl;
        exit(1);   // call system to stop
    }

    inFile >> label1 >> n >> label2 >> m;

    *pn = n; *pm = m;

    vector<vector<double> > price(n, vector<double> (m));

    cout.precision(10);
    inFile >> temp;

    printf("n = %d, m = %d \n", n,m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
          inFile >> temp;
          price[i][j] = atof(temp);
        }
    }

		inFile.close();
	  return price;
}

vector< vector<double> > getReturns(vector<vector<double> > matrix, int m, int n) {


	vector< vector<double> > returns(n, vector<double> (m));
	for (int i = 0; i < n; i ++) {
		for (int j = 0; j < m; j ++) {
			if (j == 0)
				returns[i][j] = 0;
			else
				returns[i][j] = (matrix[i][j] - matrix[i][j - 1])/ matrix[i][j - 1];
		}
	}
  return returns;
}

vector< vector<double> > dot(vector<vector<double> > v1, vector<vector<double> > v2){
  if(v1[0].size()!=v2.size()) cout << "Matrix Size mismatch error " << endl;

  vector<vector<double> > result(v1.size(), vector<double>(v2[0].size()));
	for(unsigned i = 0; i < v1.size(); ++i) {
		for(unsigned j = 0; j < v2[0].size(); ++j){
			for(unsigned k=0; k<v1[0].size(); ++k){
				result[i][j] += v1[i][k] * v2[k][j];
			}
		}
	}
  return result;
}

vector<vector<double> > add_subtract(vector<vector<double> > v1, vector<vector<double> > v2, int a){ //a=1 : add, a=-1 : subtract
    vector<vector<double> > result(v1.size(), vector<double>(v1[0].size()));
    if (v1.size() != v2.size() or v1[0].size() != v2[0].size())
        cout << "Matrix Subtraction Size mismatch error " << endl;
    for (unsigned n = 0; n < v1.size(); ++n){
        for (unsigned m=0; m<v1[0].size(); ++m){
            if(a==1) result[n][m]=v1[n][m]+v2[n][m];
            else result[n][m]=v1[n][m]-v2[n][m];
        }
    }
    return result;
}

vector<vector<double> > multiply(vector<vector<double> > v1, vector<vector<double> > v2){
    vector<vector<double> > result(v1.size(), vector<double>(v1[0].size()));
    if (v1.size() != v2.size() or v1[0].size() != v2[0].size())
        cout << "Matrix Size mismatch error " << endl;
    for (unsigned n = 0; n < v1.size(); ++n){
        for (unsigned m=0; m<v1[0].size(); ++m){
            result[n][m]=v1[n][m]*v2[n][m];
        }
    }
    return result;
}

vector<vector<double> > transpose(vector<vector<double> > v1){
    vector<vector<double> > result(v1[0].size(), vector<double>(v1.size()));
    for (unsigned n = 0; n < v1.size(); ++n){
        for (unsigned m=0; m<v1[0].size(); ++m){
            result[m][n]=v1[n][m];
        }
    }
    return result;
}

double getError(vector<vector<double> > v1, vector<vector<double> > v2){
    double sum = 0;
    double error;
    if (v1.size() != v2.size() or v1[0].size() != v2[0].size())
        cout << "Matrix Size mismatch error " << endl;
    for (unsigned n = 0; n < v1.size(); ++n){
        for (unsigned m=0; m<v1[0].size(); ++m){
            sum +=(v1[n][m]-v2[n][m])*(v1[n][m]-v2[n][m]);
        }
    }
    error = sum/(v1.size()*v1[0].size());
    return error;
}
