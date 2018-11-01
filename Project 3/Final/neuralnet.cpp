#include<iostream>
#include<fstream>
#include <vector>
#include <math.h>
#include <random>
#include<string>
#include "methods.h"

using namespace std;

vector< vector<double> > getReturns(vector<vector<double> > , int, int);
vector< vector<double> >readPrices(char *, int *, int *);

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

class NeuronLayer{
    int num_neurons;
    int num_inputs_per_neuron;
public:
    vector<vector<double> > arc_weight;
    NeuronLayer(int neurons_num, int inputs_num, mt19937 gen){
        normal_distribution<double> d(0,1);
        num_neurons = neurons_num;
        num_inputs_per_neuron = inputs_num;

        for (unsigned i =0; i < num_neurons; ++i){
            vector<double> temp;
            for (unsigned j =0; j < num_inputs_per_neuron; ++j){
                temp.push_back(d(gen) *2.0 - 1.0);
            }
            arc_weight.push_back(temp);
        }
    }
};

class NeuralNetwork{
    int num_inputs;
    int num_outputs;
    int num_hidden;
    int neurons_per_hidden;
    vector<NeuronLayer> layers;
public:
    NeuralNetwork(int a, int b, int c, int d, mt19937 gen){
        int num_inputs=a;
        int num_outputs=b;
        int num_hidden=c;
        int neurons_per_hidden=d;

        if (num_hidden>0){
            layers.push_back(NeuronLayer(num_inputs,neurons_per_hidden, gen));
            for (unsigned i =0; i < num_hidden; ++i){
                layers.push_back(NeuronLayer(neurons_per_hidden, neurons_per_hidden,gen));
            }
            layers.push_back(NeuronLayer(neurons_per_hidden,num_outputs, gen));
        }
        else //no hidden layer
            layers.push_back(NeuronLayer(num_outputs, num_inputs, gen));
    }

  double sigmoid(double x) { return 1 / (1+exp(-x)); }
	double dsigmoid(double x) { return x * (1-x); }
	vector<vector<double> > sigmoid(vector<vector<double> > x){
      vector<vector<double> > temp(x.size(), vector<double> (x[0].size()));
      for (unsigned i=0; i<x.size(); ++i){
          for (unsigned j=0; j<x[0].size(); ++j){
              temp[i][j] = sigmoid(x[i][j]);
          }
      }
      return temp;
	}

	vector<vector<double> > dsigmoid(vector<vector<double> > x){
      vector<vector<double> > temp(x.size(), vector<double> (x[0].size()));
      for (unsigned i=0; i<x.size(); ++i){
          for (unsigned j=0; j<x[0].size(); ++j){
              temp[i][j] = dsigmoid(x[i][j]);
          }
      }
      return temp;
	}

	void gradient_descent(vector<vector<double> > train_input, vector<vector<double> > validation, int loops);
  vector<vector<vector<double> > > gradient_descent(vector<vector<double> > train_input);
	vector<vector<vector<double> > > compute_output(vector<vector<double> > inputs);
};

vector<vector<vector<double> > > NeuralNetwork::compute_output(vector<vector<double> > inputs){
    vector<vector<vector<double> > > layers_out(3);
    layers_out[0] = (sigmoid(dot(transpose(inputs), layers[0].arc_weight)));
    layers_out[1] = (sigmoid(dot(layers_out[0], transpose(layers[1].arc_weight))));
    layers_out[2] = (sigmoid(dot(layers_out[1], layers[2].arc_weight)));

    return layers_out;
}

void NeuralNetwork::gradient_descent(vector<vector<double> > train_input, vector<vector<double> > validation, int loops){
    vector<vector<vector<double> > > layers_out;
    double current_error = 0;
    double new_error;
    for (unsigned n = 0; n < loops; ++n){
        layers_out = compute_output(train_input);
        vector<vector<vector<double> > > delta;
        vector<vector<vector<double> > > descent(3);
        //transpose validation?
        vector<vector<double> > temp = add_subtract(transpose(validation), layers_out[2],-1);
        delta.insert(delta.begin(),multiply(temp,dsigmoid(layers_out[2])));

        temp = dot(delta[0],transpose(layers[2].arc_weight));
        delta.insert(delta.begin(),multiply(temp,dsigmoid(layers_out[1])));

        temp = dot(delta[0],transpose(layers[1].arc_weight));
        delta.insert(delta.begin(),multiply(temp,dsigmoid(layers_out[0])));

        descent[0] = dot(train_input,delta[0]);
        descent[1] = dot(transpose(layers_out[0]),delta[1]);
        descent[2] = dot(transpose(layers_out[1]),delta[2]);

        layers[0].arc_weight = add_subtract(layers[0].arc_weight, descent[0], 1);
        layers[1].arc_weight = add_subtract(layers[1].arc_weight, descent[1], 1);
        layers[2].arc_weight = add_subtract(layers[2].arc_weight, descent[2], 1);

        new_error = getError(transpose(validation), layers_out[2]);
        cout << "The current error is: " << new_error << endl;

        // Error Tolerance
        if((abs(current_error-new_error))<0.000001) break;
      	else{
      	    current_error = new_error;
      	}
    }
}

int main(int argc, char **argv)
{
    int retcode = 0;
    int n, m;
    //cout << "n = " << n << endl;
    if (argc != 2){
        cerr << "usage: prices filename\n" << endl; retcode = 1;
        exit(1);   // call system to stop      }
    }

    vector<vector<double> > p;
    vector<vector<double> > ret;

    p = readPrices(argv[1], &n, &m);
    ret = getReturns(p,m,n);

    vector<vector<double> > train_inputs(n, vector<double> (242));
    vector<vector<double> > train_outputs(n, vector<double> (242));

    vector<vector<double> > test_inputs(n, vector<double> (241));
    vector<vector<double> > test_outputs(n, vector<double> (241));

    for (int i = 0; i < n; ++i) {
        //cout << i;
        for (int j = 1; j < 253; ++j) {
          train_inputs[i][j-1] = ret[i][j];
          train_outputs[i][j-1] = ret[i][j+10];
        }
        for (int j = 253; j < 493; ++j) {
          test_inputs[i][j-253] = ret[i][j];
          test_outputs[i][j-253] = ret[i][j+10];
        }
    }

    random_device rd;
    mt19937 gen(rd());

    NeuralNetwork network = NeuralNetwork(n,n,1,50,gen); //num_inputs, num_outputs, num_hidden, neurons_per_hidden
    network.gradient_descent(train_inputs, train_outputs, 100);

    vector<vector<vector<double> > > test_results = network.compute_output(test_inputs);

    double error = getError(transpose(test_outputs), test_results[2]);

    cout << "Final Error on Test: " << error << endl;


    return retcode;
}
