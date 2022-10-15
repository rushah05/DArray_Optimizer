#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

template<typename T>
void read_input_file(char *filename, int n, int d, T* X, int ldx, T* Y){
    FILE *f = fopen(filename, "r");
    if(!f) exit(EXIT_FAILURE);
    std::string colon(":");

    size_t read, len=0;
    char *line;
    int i=0;

    for(;i<n;++i){
        if((read=getline(&line, &len, f))!=-1){
            Y[i] = atof(strtok(line," "));
            if(Y[i] == 1) Y[i] = 1;
            else Y[i] = -1;
            char *features = strtok (NULL," ");
            while (features != NULL){
                std::string f(features);
                int found = f.find(colon);
                int index= atoi(f.substr(0, found).c_str());
                // T value=std::stod(f.substr(found+1, strlen(f.c_str())).c_str());
                T value=atof(f.substr(found+1, strlen(f.c_str())).c_str());
                if(index-1 > -1){
                    // printf("X[%d,%d]=%f - [%d]\n", index-1, i, value, (index-1)+i*d);
                    X[(index-1)+i*d] = value;
                }
                features = strtok (NULL, " ");
            }
        }
    }
}


// template<typename T>
// void read_input_file(char *filename, long *n, long *d, T **X, T **Y){
//     FILE *f = fopen(filename, "r");
//     if(!f) exit(EXIT_FAILURE);
//     std::string colon(":");

//     size_t read, len=0;
//     char *line;
//     size_t elements, l=*n, i=0;

//     for(;i<l;++i){
//         if((read=getline(&line, &len, f))!=-1){
//             (*Y)[i] = atof(strtok(line," "));
//             if((*Y)[i] == 1) (*Y)[i] = 1.0;
//             else (*Y)[i] = -1.0;
//             char *features = strtok (NULL," ");
//             while (features != NULL){
//                 std::string f(features);
//                 int found = f.find(colon);
//                 int index= atoi(f.substr(0, found).c_str());
//                 // T value=std::stod(f.substr(found+1, strlen(f.c_str())).c_str());
//                 T value=atof(f.substr(found+1, strlen(f.c_str())).c_str());
//                 if(index-1 > -1){
//                     // if(rank==0) printf("(*X)[%d,%d]=%f - [%d]\n", index-1, i, value, (index-1)+i*d);
//                     (*X)[i* (*d)+(index-1)] = value;
//                 }
//                 features = strtok (NULL, " ");
//             }
//         }
//     }
// }