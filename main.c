#include "mlp.h"
#include "dataset.h"

#define FULL_PROGRAM 1


int main(int argc, char** argv){

    FILE* fd1 = fopen("train_quake.dat","r");
    dataset trainDataset;

    parseDataset(fd1,&trainDataset);
   
    FILE* fd2 = fopen("test_quake.dat","r");
    dataset testDataset;

    parseDataset(fd2,&testDataset);

    //normalizeDatasets(&trainDataset,&testDataset);



    if(FULL_PROGRAM){
        srand(time(NULL));
        printf("\n\n");
        int topology[4] = {trainDataset.xFeatures,2,trainDataset.yFeatures,2};


        mlp* aux;
        aux = initMLP(topology,SIGMOID);

        

        
        double minerror = 99999;
        double errorTest = 0 ;

        for (size_t x = 1; x <= 1000; x++)
        {
            for(int i = 0 ; i < trainDataset.size; i++){

            resetWdelta(aux);
            feedInput(trainDataset.xData[i],aux);
            forwardPropagation(aux);
            backpropagateError(aux,trainDataset.yData[i]);
            accumulateChange(aux);
            adjustWeights(aux);
            obtainError(aux,trainDataset.yData[i]);

            errorTest += aux->header.currError;

            }
            
            printf("Iteraration %d -> Error : %lf\n",x,errorTest/trainDataset.size);
            errorTest = 0;
        }

        

        int numero = testDataset.size;

        for(int i = 0 ; i < numero  ; i++){


            feedInput(testDataset.xData[i],aux);
            forwardPropagation(aux);
            obtainError(aux,testDataset.yData[i]);

            errorTest += aux->header.currError;

        }
        errorTest = errorTest / numero; 


        

        printNetwork(aux);

        printf("Error Test -> %lf\n", errorTest);
    }

    

    return 0;
}