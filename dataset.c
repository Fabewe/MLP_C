#include "dataset.h"

void testdataset(){
    printf("Start the test...\n");

    dataset ds;
    FILE* fd = fopen("train_quake.dat","r");

    //leerArchivo(fd,&ds);

    printf("\nxFeatures->%d\nyFeatures->%d\nsize->%d\n",ds.xFeatures,ds.yFeatures,ds.size);



    printf("\n == PRINT INPUT MATRIX ==");

    for( int i = 0 ; i < ds.size ; i++){

        for(int j = 0 ; j < ds.xFeatures ; j++){
            printf(" xM[%d][%d] - %lf",i,j,ds.xData[i][j]);
        }

        printf("\n");
    }

    printf("\n == PRINT OUTPUT MATRIX ==");

    for( int i = 0 ; i < ds.size ; i++){

        for(int j = 0 ; j < ds.yFeatures ; j++){
            printf(" yM[%d][%d] - %lf",i,j,ds.yData[i][j]);
        }

        printf("\n");
    }
}

void parseDataset(FILE* fd, dataset* dataset) {


    //Read dataset dimensions
    if (fscanf(fd, "%d %d %d\n", &(dataset->xFeatures), &(dataset->yFeatures), &(dataset->size)) != 3) {
        printf("Error al leer los tres enteros");
        exit;
    }

    printf("\na->%d\nb->%d\nc->%d\n",dataset->xFeatures,dataset->yFeatures,dataset->size);


    //Allocate features matrix memory
    dataset->xData = (double**)malloc(sizeof(double *) * dataset->size);
    for( int i = 0 ; i < dataset->size ; i++){
        dataset->xData[i] = (double*)malloc( sizeof(double) * dataset->xFeatures);
    }

    //Allocate targets matrix memory
    dataset->yData = (double**)malloc(sizeof(double *) * dataset->size);
    for( int i = 0 ; i < dataset->size ; i++){
        dataset->yData[i] = (double*)malloc( sizeof(double) * dataset->yFeatures);
    }

    for( int i = 0 ; i <  dataset->size ; i++){
        for(int j = 0 ; j <  dataset->xFeatures ; j++){
            if (fscanf(fd, "%lf ", &(dataset->xData[i][j])) != 1){
                printf("Error al leer datos - X - [%d][%d]\n",i,j);
            }
        }

        for(int j = 0 ; j <  dataset->yFeatures - 1; j++){
            if (fscanf(fd, "%lf,", &(dataset->yData[i][j])) != 1){
                printf("Error al leer datos- Y - [%d][%d]\n",i,j);
            }
            
        }
        if (fscanf(fd, "%lf\n", &(dataset->yData[i][dataset->yFeatures - 1])) != 1){
                printf("Error al leer datos - Yf - [%d]\n",i);
        }

    }

    //Close file
    fclose(fd);

}


void normalizeDatasets(dataset* datasetTrain ,dataset* datasetTest ){

    int minLimX  = -1;
    int maxLimX  = 1;

    int minLimY  = 0;
    int maxLimY  = 1;

    // Get max and min
    double minValue;
    double maxValue;

    double minValueY;
    double maxValueY;


    minValue = maxValue = datasetTrain->xData[0][0];
    minValueY = maxValueY = datasetTrain->yData[0][0];


    //Check max and min values on train dataset
    for( int i = 0 ; i <  datasetTrain->size ; i++){
        for(int j = 0 ; j <  datasetTrain->xFeatures ; j++){
            
            if(datasetTrain->xData[i][j] < minValue ){
                minValue = datasetTrain->xData[i][j];
            }

            if(datasetTrain->xData[i][j] > maxValue ){
                maxValue = datasetTrain->xData[i][j];
            }

        }

        for(int j = 0 ; j <  datasetTrain->yFeatures ; j++){

            if(datasetTrain->yData[i][j] < minValueY ){
                minValueY = datasetTrain->yData[i][j];
            }

            if(datasetTrain->yData[i][j] > maxValueY ){
                maxValueY = datasetTrain->yData[i][j];
            }
            
            
        }

    }
    

    printf("Mayor ->%lf\nMenor ->%lf\n",maxValue,minValue);
    printf("MayorY ->%lf\nMenorY ->%lf\n",maxValueY,minValueY);
    
    

    //Scaling trainDataset
    for( int i = 0 ; i <  datasetTrain->size ; i++){
        for(int j = 0 ; j <  datasetTrain->xFeatures ; j++){
            

            datasetTrain->xData[i][j] = minLimX + ( ((datasetTrain->xData[i][j] - minValue)*(maxLimX - minLimX ))/(maxValue - minValue) );
        }

        for(int j = 0 ; j <  datasetTrain->yFeatures ; j++){

            datasetTrain->yData[i][j] = minLimY + ( ((datasetTrain->yData[i][j] - minValueY)*(maxLimY - minLimY ))/(maxValueY - minValueY) );
            
        }
    }

    //Scalling testDataset
    for( int i = 0 ; i <  datasetTest->size ; i++){
        for(int j = 0 ; j <  datasetTest->xFeatures ; j++){
            

            datasetTest->xData[i][j] = minLimX + ( ((datasetTest->xData[i][j] - minValue)*(maxLimX - minLimX ))/(maxValue - minValue) );
        }

        for(int j = 0 ; j <  datasetTest->yFeatures ; j++){

            datasetTest->yData[i][j] = minLimY + ( ((datasetTest->yData[i][j] - minValue)*(maxLimY - minLimY ))/(maxValue - minValue) );
            
        }
    }

}