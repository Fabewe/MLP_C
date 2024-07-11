
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>


#define NU 0.1
#define ETA 0.2



/*

Declaration of the multilayer perceptron struct

*/

//ACTIVAION FUNCTIONS
typedef enum activationF{

    SIGMOID,
    SOFTMAX

}   activationF;

//LAYER TYPE
typedef enum layerType{

    IN,
    HIDDEN,
    OUT

}   layerType;


//HEADER
typedef struct mlpHeader {

    activationF aFunction;
    uint8_t nInputs;
    uint8_t nOutputs;
    uint8_t nLayers;
    uint8_t nNeurons;
    double currError;


}   mlpHeader;

//NEURON
typedef struct mlpNeuron {

    
    double* wList;
    double* wDelta;
    double* lastWdelta;
    double value;
    double delta;
    
    
}   mlpNeuron;

//LAYER
typedef struct mlpLayer {
    uint8_t   identifier;
    layerType type;
    mlpNeuron* neurons;

}   mlpLayer;




//MLP
typedef struct mlp {

    mlpHeader header;
    mlpLayer* layers;

}   mlp;



//Initialize with topology  { nInputs , nNeurons , nOutputs , nHiddenLayers } and activation function { SIGMOID | SOFTMAX }
mlp* initMLP(int* top , activationF type);


//Print network
void printNetwork(mlp* network);


//Feed inputs to the mlp
void feedInput(double* inputs,mlp* network);

void obtainError(mlp* network, double * outputs);

void forwardPropagation(mlp* network);

void backpropagateError(mlp* network , double* target);

void accumulateChange(mlp* network);

void adjustWeights(mlp* network);

double activateF(mlpHeader header , double value);