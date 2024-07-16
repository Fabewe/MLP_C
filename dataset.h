
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#pragma once

/*
    Declaration of the dataset struct
*/

typedef struct dataset {
    
    FILE* fd;
    char* fileName;

    int xFeatures;
    int yFeatures;
    int size;

    double** xData;
    double** yData;

}   dataset;


void parseDataset(FILE* fd, dataset* dataset);
void normalizeDatasets(dataset* datasetX ,dataset* datasetY );