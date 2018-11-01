#include<iostream>
#include<fstream>
#include <vector>
#include <math.h>
#include <random>
#include<string>
#include "methods.h"

using namespace std;

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

        // Error Tolerance
        if((abs(current_error-new_error))<0.000001) {
          cout << "Final error is: " << new_error << endl;
          break;
        }
      	else{
          cout << "The current error is: " << new_error << endl;
      	    current_error = new_error;
      	}
    }
}

int main(int argc, char **argv)
{
    int retcode = 0;
    int n, m;
    if (argc != 2){
        cerr << "usage: prices filename\n" << endl; retcode = 1;
        exit(1);   // call system to stop      }
    }

    vector<vector<double> > p;
    vector<vector<double> > ret;

    p = readPrices(argv[1], &n, &m);
    ret = getReturns(p,m,n);

    // Get train and test sets
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

    // Generate random seed
    random_device rd;
    mt19937 gen(rd());

    // Create Neural Network
    NeuralNetwork network = NeuralNetwork(n,n,1,50,gen); //num_inputs, num_outputs, num_hidden, neurons_per_hidden

    // Train
    network.gradient_descent(train_inputs, train_outputs, 10000);

    // Test
    vector<vector<vector<double> > > test_results = network.compute_output(test_inputs);

    double error = getError(transpose(test_outputs), test_results[2]);

    cout << "Final Error on Test: " << error << endl;

    return retcode;
}
