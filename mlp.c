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
                }
            }

        //OUTPUT LAYER
        }else if(layer->identifier == header.nLayers + 1){


            layer->neurons = (mlpNeuron*)malloc(sizeof(mlpNeuron) * header.nOutputs);
            

            for(int i = 0 ; i < header.nOutputs; i++ ){

                layer->neurons[i].wList = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].wDelta = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].lastWdelta = (double * )calloc(sizeof(double) * header.nInputs + 1,sizeof( double ));
                

                //INITIALIZE ZERO (BORRAR)
                layer->neurons[i].value = 0;
                layer->neurons[i].delta = 0;
                for(int j = 0 ; j < header.nNeurons + 1; j++ ){
                    layer->neurons[i].wList[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
                }
            }

        //HIDDEN LAYER
        }else{

            layer->neurons = (mlpNeuron*)malloc(sizeof(mlpNeuron) * header.nNeurons);

            for(int i = 0 ; i < header.nNeurons ; i++ ){

                layer->neurons[i].wList = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].wDelta = (double * )malloc(sizeof(double) * header.nNeurons + 1);
                layer->neurons[i].lastWdelta = (double * )calloc(sizeof(double) * header.nInputs + 1, sizeof( double ));
                

                //INITIALIZE ZERO (BORRAR)
                layer->neurons[i].value = 0;
                layer->neurons[i].delta = 0;
                for(int j = 0 ; j < header.nNeurons + 1; j++ ){
                    layer->neurons[i].wList[j] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0 ;
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


    //DEBUG
    /*
      int maxNe = 0;
      if(layer->type == IN){
        maxNe =  header.nInputs;
      }else if(layer->type == OUT){
        maxNe = header.nOutputs;
      }else{
        maxNe = header.nNeurons;
      }

    printf("CAPA %d INICIALIZADA\n",layer->identifier);
    printf("Valores : \n");
    for(int i = 0 ; i < maxNe ; i++){
        printf("%f ",layer->neurons[i].value);
    }
    printf("\n");
    */
}

mlp* initMLP(int* top , activationF type){

    mlp* ret_v;

    ret_v = (mlp*)malloc(sizeof(mlp));

    //Initialize Header
    ret_v->header.aFunction = type;
    ret_v->header.nInputs = top[0];
    ret_v->header.nNeurons = top[1];
    ret_v->header.nOutputs = top[2];
    ret_v->header.nLayers = top[3];


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

    printf("===========\nN Inputs: %u \nN Outputs: %u \nN Neurons: %u \nN Hidden Layers: %u \nActivation Function: %s \n===========\n\nLayers:\n",network->header.nInputs,network->header.nOutputs,network->header.nNeurons,network->header.nLayers,getActivationFunction(network->header.aFunction));

    //Print layers
    
    //PRINT INPUT LAYER
    printf("CAPA %d ",network->layers[0].identifier);
    printf("Valores : \n");
    for(int i = 0 ; i < network->header.nInputs ; i++){
        printf("%f ",network->layers[0].neurons[i].value);
    }
    printf("\n");


    //PRINT HIDDEN LAYERS
    for(int i = 1 ; i < network->header.nLayers + 1 ; i++ ){

        printf("CAPA %d ",network->layers[i].identifier);
        printf("Valores : \n");
        for(int j = 0 ; j < network->header.nNeurons ; j++){
            printf("%f ",network->layers[i].neurons[j].value);
        }
        printf("\n");
    }
    
    //PRINT OUTPUT LAYER
    printf("CAPA OUTPUT\n");
    printf("Valores : \n");
    for(int i = 0 ; i < network->header.nOutputs ; i++){
        printf("%f ",network->layers[network->header.nLayers + 1].neurons[i].value);
    }
    printf("\n");
}


void feedInput(double* inputs,mlp* network){

    for(int i = 0 ; i < network->header.nInputs; i++){
        network->layers[0].neurons[i].value = inputs[i];
    }

}

void forwardPropagation(mlp* network){

    double net;

    //FIRST LAYER
    for(int i = 0 ; i < network->header.nNeurons; i++){
        net = 0;
        for(int j = 1 ; j < network->header.nInputs +  1; j++){

            //printf(" %f --",network->layers[1].neurons[i].wList[j]);
            net += network->layers[1].neurons[i].wList[j] * network->layers[0].neurons[j].value;

        }

        //Add bias
        net += network->layers[1].neurons[i].wList[0];

        ///Apply the activation (sigmoide) function
        network->layers[1].neurons[i].value = activateF(network->header,net);
    }


    //HIDDEN LAYERS
    for(int  k = 2 ; k <  network->header.nLayers + 1 ; k++){

        for(int i = 0 ; i < network->header.nNeurons; i++){

            net = 0;
            for(int j = 1 ; j < network->header.nNeurons + 1; j++){
                //printf(" %f --",network->layers[k].neurons[i].wList[j]);
                net += network->layers[k].neurons[i].wList[j] * network->layers[k-1].neurons[j].value;

            }

            //Add bias
            net += network->layers[k].neurons[i].wList[0];


            network->layers[k].neurons[i].value  = activateF(network->header,net);
        }
    }

    //OUTPUT LAYER
    for(int i = 0 ; i < network->header.nOutputs; i++){
        net = 0;
        for(int j = 1 ; j < network->header.nNeurons + 1; j++){
            //printf(" %f --",network->layers[network->header.nLayers + 1].neurons[i].wList[j]);
            net += network->layers[network->header.nLayers + 1].neurons[i].wList[j] * network->layers[network->header.nLayers].neurons[j].value;

        }

        //Add bias
        net += network->layers[network->header.nLayers + 1].neurons[i].wList[0];

        network->layers[network->header.nLayers + 1].neurons[i].value  = activateF(network->header,net);
    }
}

void backpropagateError(mlp* network , double* target){

    int lastLayer = network->header.nLayers + 1;

    //For each output neuron
    for ( int j = 0; j < network->header.nOutputs; j++ ){
		//It does the formula more visual
		double dj = target[j];
		double outHj = network->layers[lastLayer].neurons[j].value;

		//-(resulObjetivo[j] - salida[j]) * g'(net^H_j) //La derivada es: salida[j]*(1-salida[j])
		network->layers[lastLayer].neurons[j].delta = - 2 * (dj - outHj) * (outHj * ( 1 - outHj ) );
	}


    //Last hidden layer
    for ( int j = 0; j < network->header.nNeurons; j++ ){

            double net = 0.0;
	 		for( int i = 0; i < network->header.nOutputs; i++ ){

				
				double weightH1ij = network->layers[lastLayer].neurons[i].wList[j+1];
				double deltaH1i = network->layers[lastLayer].neurons[i].delta;

			    net += weightH1ij * deltaH1i;
		    }
 	

        double outhj = network->layers[lastLayer-1].neurons[j].value;

 		network->layers[lastLayer-1].neurons[j].delta = net * (outhj * (1 - outhj));
    }


    
    
    //For each normal layer
    for ( int h = lastLayer - 2; h > 0; h-- ){

        for ( int j = 0; j < network->header.nNeurons; j++ ){

            double net = 0.0;
	 		for( int i = 0; i < network->header.nNeurons; i++ ){

				//It does the formula more visual
				double weightH1ij =  network->layers[h+1].neurons[i].wList[j+1];
				double deltaH1i =  network->layers[h+1].neurons[i].delta;

			    net += weightH1ij * deltaH1i;
		}
		//It does the formula more visual
 		double outhj = network->layers[h].neurons[j].value;

 		network->layers[h].neurons[j].delta = net * (outhj * (1 - outhj));
        }
	}

}


void accumulateChange(mlp* network){


    //First hidden layer
    //For each neuron of first hidden layer
	for ( int j = 0; j < network->header.nNeurons; j++ ){

		//For each neuron on input layer
        for ( int i = 1; i <= network->header.nInputs; i++ ){

		    //It does the formula more visual
            double deltaWhji = network->layers[1].neurons[j].wDelta[i];
            double deltahj = network->layers[1].neurons[i].delta;
			double outh1i = network->layers[0].neurons[i].value;

			network->layers[1].neurons[j].wDelta[i] = deltaWhji + deltahj * outh1i;
		}


        //Bias
        double deltaWhjBias = network->layers[1].neurons[j].wDelta[0];
		double deltahj = network->layers[1].neurons[j].delta;

		network->layers[1].neurons[j].wDelta[0] = deltaWhjBias + deltahj;	

	}



    // //For hidden layers
	for ( int h = 2; h <= network->header.nLayers; h++ ){

		//For each neuron of layer h
		for ( int j = 0; j < network->header.nNeurons; j++ ){
			//For each neuron of layer h-1
			for ( int i = 1; i < network->header.nNeurons; i++ ){
				//It does the formula more visual
				double deltaWhji = network->layers[h].neurons[j].wDelta[i];
				double deltahj = network->layers[h].neurons[j].delta;
				double outh1i = network->layers[h-1].neurons[i].value;

				network->layers[h].neurons[j].wDelta[i] = deltaWhji + deltahj * outh1i;
			}

			//Bias
			//It does the formula more visual
			double deltaWhjBias = network->layers[h].neurons[j].wDelta[0];
			double deltahj = network->layers[h].neurons[j].delta;

			network->layers[h].neurons[j].wDelta[0] = deltaWhjBias + deltahj;
		}
	}


    //Output layer
    // //For each neuron of layer h
    for ( int j = 0; j < network->header.nOutputs; j++ ){
	    //For each neuron of layer h-1
		for( int i = 1; i < network->header.nNeurons; i++ ){
			//It does the formula more visual
			double deltaWhji = network->layers[network->header.nLayers + 1 ].neurons[j].wDelta[i];
			double deltahj = network->layers[network->header.nLayers + 1].neurons[j].delta;
			double outh1i = network->layers[network->header.nLayers ].neurons[i].value;

			network->layers[network->header.nLayers + 1].neurons[j].wDelta[i] = deltaWhji + deltahj * outh1i;
		}

		//Bias
		//It does the formula more visual
		double deltaWhjBias =  network->layers[network->header.nLayers + 1].neurons[j].wDelta[0];
		double deltahj = network->layers[network->header.nLayers + 1].neurons[j].delta;

		network->layers[network->header.nLayers + 1].neurons[j].wDelta[0] = deltaWhjBias + deltahj;


	}
}


void adjustWeights(mlp* network){


    //First hidden layer
    for(int j = 0 ; j < network->header.nNeurons; j++){

        for(int i = 1 ; i < network->header.nInputs; i++){
            
            double aux = network->layers[1].neurons[j].wList[i];
            network->layers[1].neurons[j].wList[i] = network->layers[1].neurons[j].wList[i] - \
                                                    (NU * network->layers[1].neurons[j].wDelta[i]) - \
                                                    (ETA*(NU * network->layers[1].neurons[j].lastWdelta[i]));



            network->layers[1].neurons[j].lastWdelta[i] = aux;

        }

        double aux = network->layers[1].neurons[j].wList[0];
        network->layers[1].neurons[j].wList[0] = network->layers[1].neurons[j].wList[0] - \
                                                    (NU * network->layers[1].neurons[j].wDelta[0]) - \
                                                    (ETA*(NU * network->layers[1].neurons[j].lastWdelta[0]));

        network->layers[1].neurons[j].lastWdelta[0] = aux;
    }



    //for(int h = 1 ; h < network->header.nLayers; h++){}


}


void obtainError(mlp* network, double * outputs){

    double total = 0 ;
    //MSE ERROR

    for(int i = 0 ; i < network->header.nOutputs ; i++){
        total += pow(network->layers[network->header.nLayers + 1].neurons[i].value - outputs[i],2);
    }

    //printf("\nError1 = %f",total);
    //printf("\nError2= %f",total/network->header.nOutputs);

    network->header.currError =  total/network->header.nOutputs;

}

double activateF(mlpHeader header , double value){

    double ret_val = 0;

    if(header.aFunction == SIGMOID){
        ret_val = 1 / (1 + exp(-value));
    }


    return ret_val;

}