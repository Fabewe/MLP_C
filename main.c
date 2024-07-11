#include "mlp.h"



void main(int argc, char** argv){

    srand(time(NULL));

    printf("\n\n");
    int topology[4] = {2,4,4,5};
    double input[2] = {1,2};

    double output[4] = { 3, 2, 1,4};
    // double * output;

    // output = (double  *)malloc(sizeof(double) * 4);

    // output[0] = 3;
    // output[1] = 2;
    // output[2] = 1;
    // output[3] = 4;

    mlp* aux;
    aux = initMLP(topology,SIGMOID);


    
    for(int i = 0 ; i< 25 ; i++){
        feedInput(input,aux);
        forwardPropagation(aux);
        obtainError(aux,output);
        backpropagateError(aux,output);
        accumulateChange(aux);
        adjustWeights(aux);
        printf("Iter  = %d || Error -> %f\n", i + 1,aux->header.currError);
    }

    obtainError(aux,output);
    //printNetwork(aux);
    
    printf("Error = %f\n", aux->header.currError);

    //printf("\nEn principio todo bien\n\n");
}


