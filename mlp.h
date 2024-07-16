
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#pragma once

#define NU 0.5
#define ETA 0.1



/*
    Declaration of the multilayer perceptron structs
*/

//ACTIVATION FUNCTIONS
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
    double eta;
    double mu;
    double currError;


}   mlpHeader;

//NEURON
typedef struct mlpNeuron {

    
    double* wList;
    double* wDelta;
    double* lastWdelta;
    double value;
    double net;
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
mlp* initMLP(int* top , activationF type , double eta , double nu);


//Print network
void printNetwork(mlp* network);


//Feed inputs to the mlp
void feedInput(double* inputs,mlp* network);


//Obtain error and save on network header
void obtainError(mlp* network, double * outputs);

//Forward propagate
void forwardPropagation(mlp* network);

//Backprogation and set all deltas
void backpropagateError(mlp* network , double* target);

//Accumulate change on wDelta list
void accumulateChange(mlp* network);

//Weights adjustments
void adjustWeights(mlp* network);

//Makes all wDelta list 0
void resetWdelta(mlp* network);

//Returns the value of the activation function
double activateF(mlpHeader header , double value);