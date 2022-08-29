#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

void read_input_file(int rank, char *filename, long long int gn, int d, float* X, long long int ldx, float* Y){
    FILE *f = fopen(filename, "r");
    if(!f) exit(EXIT_FAILURE);
    std::string colon(":");

    size_t read, len=0;
    char *line;
    long long int i=0;

    for(;i<gn;++i){
        if((read=getline(&line, &len, f))!=-1){
            Y[i] = atof(strtok(line," "));
            // if(Y[i] == 1) Y[i] = 1;
            // else Y[i] = -1;
            char *features = strtok (NULL," ");
            while (features != NULL){
                std::string f(features);
                int found = f.find(colon);
                int index= atoi(f.substr(0, found).c_str());
                float value=atof(f.substr(found+1, strlen(f.c_str())).c_str());
                if(index-1 > -1){
                    // if(rank==0) printf("X[%d,%d]=%f - [%d]\n", index-1, i, value, (index-1)+i*d);
                    X[(index-1)+i*d] = value;
                }
                features = strtok (NULL, " ");
            }
        }
    }
}