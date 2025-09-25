#include "mlp.h"

void layerInit(mlpLayer* layer , mlpHeader header){

    //HIDDEN LAYER
    if(layer->type != IN){

    
        //FIRST HIDDEN LAYER
        if(layer->identifier == 1){

            layer->neurons = (mlpNeuron*)malloc(sizeof(mlpNeuron) * header.nNeurons);

            for(int i = 0 ; i < header.nNeurons ; i++ ){

                layer->neurons[i].wList = (double * )malloc(sizeof(double) * header.nInputs + 1);
                layer->neurons[i].wDelta = (double * )malloc(sizeof(double) * header.nInputs + 1);
                layer->neurons[i].lastWdelta = (double * )calloc(sizeof(double) * header.nInputs + 1,sizeof( double ));
                


                //INITIALIZE ZERO (BORRAR)
                layer->neurons[i].value = 0;
                layer->neurons[i].delta = 0;
                for(int j = 0 ; j < header.nInputs + 1 ; j++ ){

                    layer->neurons[i].wList[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                    layer->neurons[i].wDelta[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                }
                
            }

        //OUTPUT LAYER
        }else if(layer->identifier == header.nLayers + 1){


            layer->neurons = (mlpNeuron*)malloc(sizeof(mlpNeuron) * header.nOutputs);
            

            for(int i = 0 ; i < header.nOutputs; i++ ){

                layer->neurons[i].wList = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].wDelta = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].lastWdelta = (double * )calloc(sizeof(double) * header.nNeurons + 1,sizeof( double ));
                

                //INITIALIZE ZERO (BORRAR)
                layer->neurons[i].value = 0;
                layer->neurons[i].delta = 0;
                for(int j = 0 ; j < header.nNeurons + 1; j++ ){
                    layer->neurons[i].wList[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                    layer->neurons[i].wDelta[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                }
            }

        //HIDDEN LAYER
        }else{

            layer->neurons = (mlpNeuron*)malloc(sizeof(mlpNeuron) * header.nNeurons);

            for(int i = 0 ; i < header.nNeurons ; i++ ){

                layer->neurons[i].wList = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].wDelta = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].lastWdelta = (double * )calloc(sizeof(double) * header.nNeurons + 1, sizeof( double ));
                

                //INITIALIZE ZERO (BORRAR)
                layer->neurons[i].value = 0;
                layer->neurons[i].delta = 0;
                for(int j = 0 ; j < header.nNeurons + 1; j++ ){
                    layer->neurons[i].wList[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                    layer->neurons[i].wDelta[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                }
            }
        }        

        

    //INPUT LAYER
    }else{
        
        layer->neurons = (mlpNeuron*)malloc(sizeof(mlpNeuron) * header.nInputs);

        //INITIALIZE ZERO (BORRAR)
            for(int i = 0 ; i < header.nInputs ; i++ ){
                layer->neurons[i].value = 0;
                layer->neurons[i].delta = 0;
            }
    }

}

mlp* initMLP(int* top , activationF type , double eta , double mu){

    mlp* ret_v;

    ret_v = (mlp*)malloc(sizeof(mlp));

    //Initialize Header
    ret_v->header.aFunction = type;
    ret_v->header.nInputs = top[0];
    ret_v->header.nNeurons = top[1];
    ret_v->header.nOutputs = top[2];
    ret_v->header.nLayers = top[3];
    ret_v->header.eta = eta;
    ret_v->header.mu = mu;


    //Initialize layers
    ret_v->layers = (mlpLayer*)malloc( sizeof(mlpLayer) * (ret_v->header.nLayers + 2) );



    //INPUTLAYER
    ret_v->layers[0].identifier = 0;
    ret_v->layers[0].type = IN;
    layerInit(&(ret_v->layers[0]),ret_v->header);

    //HIDDEN LAYERS
    for(int i = 1; i<ret_v->header.nLayers + 1 ; i++){
        ret_v->layers[i].identifier = i;
        ret_v->layers[i].type = HIDDEN;
        layerInit(&(ret_v->layers[i]),ret_v->header);
        
    }

    //OUTPUT LAYER
    ret_v->layers[ret_v->header.nLayers + 1].identifier = ret_v->header.nLayers + 1;
    ret_v->layers[ret_v->header.nLayers + 1].type = OUT;
    layerInit(&(ret_v->layers[ret_v->header.nLayers + 1]),ret_v->header);

    return ret_v;
}

const char* getActivationFunction(activationF function){

    switch (function){
      case SIGMOID: return "Sigmoid";
      case SOFTMAX: return "Softmax";

   }

}

void printNetwork(mlp* network){

    
    //Print mlp data
    printf("===========\nN Inputs: %u \nN Outputs: %u \nN Neurons: %u \nN Hidden Layers: %u \nActivation Function: %s \neta->%lf\nmu->%lf\n===========\n\nLayers:\n",network->header.nInputs,network->header.nOutputs,network->header.nNeurons,network->header.nLayers,getActivationFunction(network->header.aFunction),network->header.eta,network->header.mu);

    //Print layers
    
    //PRINT INPUT LAYER
    printf("CAPA INPUT\n");
    printf("Valores : \n");
    for(int i = 0 ; i < network->header.nInputs ; i++){
        printf("%lf ",network->layers[0].neurons[i].value);
    }
    printf("\n");


    if(network->header.nLayers>0){
        printf("CAPA %d\n",1);
        for(int j = 0 ; j < network->header.nNeurons ; j++){
                printf(" %lf ->",network->layers[1].neurons[j].value);
                for(int x = 0 ; x < network->header.nInputs ; x++){
                    printf(" %lf",network->layers[1].neurons[j].wList[x]);

                }
                printf("||");
        }
    }
     printf("\n");
    //PRINT HIDDEN LAYERS
    for(int i = 2 ; i <= network->header.nLayers  ; i++ ){

        printf("CAPA %d ",network->layers[i].identifier);
        printf("Valores : \n");
        for(int j = 0 ; j < network->header.nNeurons ; j++){
            printf("%lf ->",network->layers[i].neurons[j].value);
            for(int x = 0 ; x < network->header.nNeurons +1; x++){
                printf(" %lf",network->layers[i].neurons[j].wList[x]);

            }
            printf("||");

        }
        printf("\n");
    }
    
    //PRINT OUTPUT LAYER
    printf("CAPA OUTPUT\n");
    printf("Valores : \n");
    for(int i = 0 ; i < network->header.nOutputs ; i++){
        printf("%lf ->",network->layers[network->header.nLayers + 1].neurons[i].value);
        for(int x = 0 ; x < network->header.nNeurons +1; x++){
                printf(" %lf",network->layers[network->header.nLayers + 1].neurons[i].wList[x]);

            }
    }
    printf("\n");
}


void feedInput(double* inputs,mlp* network){

    for(int i = 0 ; i < network->header.nInputs; i++){
        network->layers[0].neurons[i].value = inputs[i];
    }

}

void forwardPropagation(mlp* network){

    double net = 0;

    //FIRST LAYER
    for(int i = 0 ; i < network->header.nNeurons; i++){
        net = 0;
        for(int j = 0 ; j < network->header.nInputs; j++){

            net += network->layers[1].neurons[i].wList[j+1] * network->layers[0].neurons[j].value;

        }

        //Add bias
        net += network->layers[1].neurons[i].wList[0];

        ///Apply the activation function
        network->layers[1].neurons[i].value = activateF(network->header,net);
        network->layers[1].neurons[i].net = net;

    }


    //HIDDEN LAYERS
    for(int  k = 2 ; k <=  network->header.nLayers  ; k++){

        for(int i = 0 ; i < network->header.nNeurons; i++){

            net = 0;
            for(int j = 0 ; j < network->header.nNeurons ; j++){
                net += network->layers[k].neurons[i].wList[j+1] * network->layers[k-1].neurons[j].value;

            }

            //Add bias
            net += network->layers[k].neurons[i].wList[0];

            ///Apply the activation function
            network->layers[k].neurons[i].value  = activateF(network->header,net);
            network->layers[k].neurons[i].net = net;
        }
    }

    //OUTPUT LAYER
    for(int i = 0 ; i < network->header.nOutputs; i++){
        net = 0;
        for(int j = 0 ; j < network->header.nNeurons + 1; j++){

            net += network->layers[network->header.nLayers + 1].neurons[i].wList[j+1] * network->layers[network->header.nLayers].neurons[j].value;

        }

        //Add bias
        net += network->layers[network->header.nLayers + 1].neurons[i].wList[0];

        ///Apply the activation function
        network->layers[network->header.nLayers + 1].neurons[i].value  =   activateF(network->header,net);
        network->layers[network->header.nLayers + 1].neurons[i].net = net;

    }
}

void backpropagateError(mlp* network , double* target){

    int lastLayer = network->header.nLayers + 1;    //Output layer index

    //For each output neuron
    for ( int j = 0; j < network->header.nOutputs; j++ ){

		double dj = target[j];
		double outHj = network->layers[lastLayer].neurons[j].value;

		network->layers[lastLayer].neurons[j].delta = - (dj - outHj) * outHj * ( 1 - outHj ) ;
	}


    //LAST HIDDEN LAYER
    for ( int j = 0; j < network->header.nNeurons; j++ ){

            double net = 0.0;
	 		for( int i = 0; i < network->header.nOutputs; i++ ){

				
				double weightH1ij = network->layers[lastLayer].neurons[i].wList[j+1];
				double deltaH1i = network->layers[lastLayer].neurons[i].delta;
                

			    net += weightH1ij * deltaH1i;
		    }
 	
           

        double outhj = network->layers[lastLayer-1].neurons[j].value;
    
 		network->layers[lastLayer-1].neurons[j].delta =  net * outhj * ( 1 - outhj);
    }


    
    
    //HIDDEN LAYERS
    for ( int h = lastLayer - 2; h > 0; h-- ){
        for ( int j = 0; j < network->header.nNeurons; j++ ){

            double net = 0.0;
	 		for( int i = 0; i < network->header.nNeurons; i++ ){


				double weightH1ij =  network->layers[h+1].neurons[i].wList[j+1];
				double deltaH1i =  network->layers[h+1].neurons[i].delta;

			    net += weightH1ij * deltaH1i;
		}

 		double outhj = network->layers[h].neurons[j].value;

 		network->layers[h].neurons[j].delta =  net * outhj * (1 - outhj);
        }
	}

}


void accumulateChange(mlp* network){


    //FIRST HIDDEN LAYER
	for ( int j = 0; j < network->header.nNeurons; j++ ){

		//For each neuron on input layer
        for ( int i = 1; i <= network->header.nInputs; i++ ){

		    //It does the formula more visual
            double deltahj = network->layers[1].neurons[j].delta;
			double outh1i = network->layers[0].neurons[i - 1].value;

			network->layers[1].neurons[j].wDelta[i] += (deltahj * outh1i);
		}


        //Bias
        double deltaWhjBias = network->layers[1].neurons[j].wDelta[0];
		double deltahj = network->layers[1].neurons[j].delta;
		network->layers[1].neurons[j].wDelta[0] = deltaWhjBias + deltahj;	

	}



    //HIDDEN LAYERS
	for ( int h = 2; h <= network->header.nLayers; h++ ){

		//For each neuron of layer h
		for ( int j = 0; j < network->header.nNeurons; j++ ){
			//For each neuron of layer h-1
			for ( int i = 1; i <= network->header.nNeurons; i++ ){
				//It does the formula more visual
				double deltahj = network->layers[h].neurons[j].delta;
				double outh1i = network->layers[h-1].neurons[i-1].value;

				network->layers[h].neurons[j].wDelta[i] += (deltahj * outh1i);
			}

			//Bias
			//It does the formula more visual
			double deltahj = network->layers[h].neurons[j].delta;

			network->layers[h].neurons[j].wDelta[0] += deltahj;
		}
	}


    //OUTPUT LAYER
    for ( int j = 0; j < network->header.nOutputs; j++ ){
	    //For each neuron of layer h-1
		for( int i = 1; i <= network->header.nNeurons; i++ ){


			double deltahj = network->layers[network->header.nLayers + 1].neurons[j].delta;
			double outh1i = network->layers[network->header.nLayers ].neurons[i-1].value;

			network->layers[network->header.nLayers + 1].neurons[j].wDelta[i] += deltahj * outh1i;
		}


		//Add bias
		double deltahj = network->layers[network->header.nLayers + 1].neurons[j].delta;

		network->layers[network->header.nLayers + 1].neurons[j].wDelta[0] += deltahj;


	}
}


void adjustWeights(mlp* network){


    //FIRST HIDDEN LAYER
    for(int j = 0 ; j < network->header.nNeurons; j++){

        for(int i = 1 ; i <= network->header.nInputs; i++){
            
            double aux = network->layers[1].neurons[j].wList[i];
            network->layers[1].neurons[j].wList[i] = aux - (network->header.mu * network->layers[1].neurons[j].wDelta[i])  \
                                                    - (network->header.eta *(network->header.mu * network->layers[1].neurons[j].lastWdelta[i]));
            network->layers[1].neurons[j].lastWdelta[i] =  network->layers[1].neurons[j].wList[i];

        }

        double aux = network->layers[1].neurons[j].wList[0];
        network->layers[1].neurons[j].wList[0] = aux - (network->header.mu * network->layers[1].neurons[j].wDelta[0])  \
                                                - (network->header.eta*(network->header.mu * network->layers[1].neurons[j].lastWdelta[0]));

        network->layers[1].neurons[j].lastWdelta[0] =  network->layers[1].neurons[j].wList[0];
    }



    //HIDDEN LAYERS
    for(int h = 2 ; h <= network->header.nLayers ; h++){
    
        for(int j = 0 ; j < network->header.nNeurons; j++){

            for(int i = 1 ; i <= network->header.nNeurons; i++){
                
                double aux = network->layers[h].neurons[j].wList[i];
                network->layers[h].neurons[j].wList[i] = aux - (network->header.mu * network->layers[h].neurons[j].wDelta[i]) \
                                                         - (network->header.eta*(network->header.mu * network->layers[h].neurons[j].lastWdelta[i]));

                network->layers[h].neurons[j].lastWdelta[i] =   network->layers[h].neurons[j].wList[i];

            }

        double aux = network->layers[h].neurons[j].wList[0];
        network->layers[h].neurons[j].wList[0] = aux - (network->header.mu * network->layers[h].neurons[j].wDelta[0]) \
                                                - (network->header.eta*(network->header.mu * network->layers[h].neurons[j].lastWdelta[0]));

        network->layers[h].neurons[j].lastWdelta[0] = network->layers[h].neurons[j].wList[0];
        }
    }

    //OUTPUT LAYER
    int outLayer = network->header.nLayers + 1; //Output layer index
    for(int j = 0 ; j < network->header.nOutputs; j++){

            for(int i = 1 ; i <= network->header.nNeurons; i++){
                
                double aux = network->layers[outLayer].neurons[j].wList[i];
                network->layers[outLayer].neurons[j].wList[i] = aux - (network->header.mu * network->layers[outLayer].neurons[j].wDelta[i]) \
                                                                - (network->header.eta*(network->header.mu * network->layers[outLayer].neurons[j].lastWdelta[i]));



                network->layers[outLayer].neurons[j].lastWdelta[i] =  network->layers[outLayer].neurons[j].wList[i];

            }

        double aux = network->layers[outLayer].neurons[j].wList[0];
        network->layers[outLayer].neurons[j].wList[0] = network->layers[outLayer].neurons[j].wList[0] - (network->header.mu * network->layers[outLayer].neurons[j].wDelta[0]) -(network->header.eta*(network->header.mu * network->layers[outLayer].neurons[j].lastWdelta[0]));

        network->layers[outLayer].neurons[j].lastWdelta[0] =  network->layers[outLayer].neurons[j].wList[0];
    }


}

void resetWdelta(mlp* network){

    //FIRST HIDDEN LAYER
    for(int j = 0 ; j < network->header.nNeurons; j++){

        for(int i = 0 ; i <= network->header.nInputs; i++){
            
            network->layers[1].neurons[j].wDelta[i] = 0;
        }
    }



    //HIDDEN LAYERS
    for(int h = 2 ; h <= network->header.nLayers ; h++){
    
        for(int j = 0 ; j < network->header.nNeurons; j++){

            for(int i = 1; i <= network->header.nNeurons; i++){
                
                network->layers[h].neurons[j].wDelta[i] = 0;

            }

        }
    }

    //OUTPUT LAYER
    int outLayer = network->header.nLayers + 1;
    for(int j = 0 ; j < network->header.nOutputs; j++){

            for(int i = 0 ; i <= network->header.nNeurons; i++){
                
                network->layers[outLayer].neurons[j].wDelta[i] = 0;

            }

    }


}

void obtainError(mlp* network, double * outputs){

    double total = 0.0 ;

    //MSE ERROR
    for(int i = 0 ; i < network->header.nOutputs ; i++){
        //printf("Comparando %lf - %lf \n",network->layers[network->header.nLayers + 1].neurons[i].value,outputs[i]);
        total += pow((network->layers[network->header.nLayers + 1].neurons[i].value - outputs[i]),2);
    }


    network->header.currError =  total/ (double)network->header.nOutputs;

}

double activateF(mlpHeader header , double value){

    double ret_val = 0;


    switch (header.aFunction){

        case SIGMOID:
                ret_val = 1 / (1 + exp(-value)); // f(x) = 1 / (1 + e^-x)
            break;
        
        default:
            ret_val = 0;
            break;
    }



    return ret_val;
}


int exportMLP(const char* exportFile, mlp* exportMLP){

    FILE *fexp = fopen(exportFile,"WB");

    //fwrite(&(exportMLP->header),sizeof(exportMLP->header),1,fexp);

    mlpHeader auxHeader = exportMLP->header;

    //fwrite(&(auxHeader),sizeof(auxHeader),1,fexp);

    printf("\n%d Size  = %ld",auxHeader.aFunction,sizeof(auxHeader.aFunction));
    printf("\n%d Size  = %ld",auxHeader.nInputs,sizeof(auxHeader.nInputs));
    printf("\n%d Size  = %ld",auxHeader.nOutputs,sizeof(auxHeader.nOutputs));
    printf("\n%d Size  = %ld",auxHeader.nLayers,sizeof(auxHeader.nLayers));
    printf("\n%d Size  = %ld",auxHeader.nNeurons,sizeof(auxHeader.nNeurons));
    printf("\n%f Size  = %ld",auxHeader.eta,sizeof(auxHeader.eta));
    printf("\n%f Size  = %ld",auxHeader.mu,sizeof(auxHeader.mu));
    printf("\n%f Size  = %ld",auxHeader.currError,sizeof(auxHeader.currError));
    printf("\nSize  = %ld",sizeof(exportMLP->header));

    printf("\n");



    fclose(fexp);
    return 0;
}
