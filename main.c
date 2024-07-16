/*
 
     This is an example of how the mlp/dataset library is intended to be used
 
*/

#include "mlp.h"
#include "dataset.h"

#include <stdbool.h>




int main(int argc, char **argv){    srand(time(NULL));
    

 /*
    ======================
     Datasets parameters
    ======================
 */

    //File paths
    const char *testFile = "./datasets/test_quake.dat";
    const char *trainFile = "./datasets/train_quake.dat";

    //Normalize dataset ?
    bool is_normalized = false;


 /*
    ======================
     MLP parameters
    ======================
 */

    int nIterations     = 2;                   // Number of iterations
    double mlpEta       = 0.1;                  // Eta hiperparameter for training
    double mlpMu        = 0.5;                  // Mu hiperparameter for training
    int nHiddenLayers   = 3;                    // Number of hidden layers
    int nHiddenNeurons  = 2;                    // Number of neurons on each layer
    activationF actF    = SIGMOID;              //Activation function (hidden + output layers)



/*
    ======================
     Usage
    ======================
 */


    FILE *fd1 = fopen(trainFile, "r");

    if (fd1 == NULL){
        printf("Error reading training dataset");
        return 1;
    }



    FILE *fd2 = fopen(testFile, "r");

    if (fd2 == NULL){
        printf("Error reading test dataset");
        return 1;
    }

    dataset trainDataset;
    dataset testDataset;


    //Initialize datasets
    parseDataset(fd1, &trainDataset);
    parseDataset(fd2, &testDataset);


    //Normalize datasets
    if(is_normalized) normalizeDatasets(&trainDataset, &testDataset);

    
    

    

    // Initialization of the mlp

    //Topology array structure : { nInputs , nNeurons , nOutputs , nHiddenLayers }
    int topology[4] = {trainDataset.xFeatures, nHiddenNeurons, trainDataset.yFeatures, nHiddenLayers};
    
    mlp * aux = initMLP(topology, actF , mlpEta , mlpMu );



    double minerror = 99999;
    double errorTest = 0;
    for (int x = 1; x <= nIterations; x++){
        for (int i = 0; i < trainDataset.size; i++){
            resetWdelta(aux);
            feedInput(trainDataset.xData[i], aux);
            forwardPropagation(aux);
            backpropagateError(aux, trainDataset.yData[i]);
            accumulateChange(aux);
            adjustWeights(aux);
            obtainError(aux, trainDataset.yData[i]);
            errorTest += aux->header.currError;
        }
        printf("Ieration %d -> Error : %lf\n",x, (errorTest/trainDataset.size));
        errorTest = 0;
    }
    int numero = testDataset.size;
    for (int i = 0; i < numero; i++)
    {
        feedInput(testDataset.xData[i], aux);
        forwardPropagation(aux);
        obtainError(aux, testDataset.yData[i]);
        errorTest += aux->header.currError;
    }
    errorTest = errorTest / numero;
    printNetwork(aux);
    printf("Error Test -> %lf\n", errorTest);
return 0;
}