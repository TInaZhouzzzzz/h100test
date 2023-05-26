#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include <iomanip>      // std::setprecision

#include<cuda_fp16.h>
#include "include/half.hpp"
using namespace std;
union FP32{
    unsigned short int f;
    unsigned int i;
};

__global__ void test(unsigned short int* A, unsigned short int* B, unsigned short int* C, unsigned short int* D){
    asm("fma.rn.bf16  %0, %1, %2, %3;" : "=h"(D[0]) : "h"(A[0]) , "h"(B[0]) , "h"(C[0]));
}

int main(int argc, char** argv){
    int size = 1;
    unsigned short int *dataA = (unsigned short int*)malloc(sizeof(unsigned short int) * size);
    unsigned short int *dataB = (unsigned short int*)malloc(sizeof(unsigned short int) * size);
    unsigned short int *dataC = (unsigned short int*)malloc(sizeof(unsigned short int) * size);
    unsigned short int *dataD = (unsigned short int*)malloc(sizeof(unsigned short int) * size);
    unsigned short int *d_dataA = NULL;
    unsigned short int *d_dataB = NULL;
    unsigned short int *d_dataC = NULL;
    unsigned short int *d_dataD = NULL;
    cudaMalloc((void**)&d_dataA, sizeof(unsigned short int) * size);
    cudaMalloc((void**)&d_dataB, sizeof(unsigned short int) * size);
    cudaMalloc((void**)&d_dataC, sizeof(unsigned short int) * size);
    cudaMalloc((void**)&d_dataD, sizeof(unsigned short int) * size);

    dataA[size-1]  = 0x5d84;
    dataB[size-1]  = 0xa230;
    dataC[size-1]  = 0x0100;


    for(int i=0;i<size;i++){
        dataD[i] = 0;
    }
    cudaMemcpy(d_dataA,dataA,sizeof(unsigned short int) * size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB,dataB,sizeof(unsigned short int) * size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataC,dataC,sizeof(unsigned short int) * size,cudaMemcpyHostToDevice);
    test<<<1, 1>>> (d_dataA, d_dataB, d_dataC, d_dataD);
    cudaMemcpy(dataD,d_dataD,sizeof(unsigned short int) * size, cudaMemcpyDeviceToHost);

    std::cout <<std::hex <<  dataD[size-1] << std::endl;

    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);
    cudaFree(d_dataD);
    free(dataA);
    free(dataB);
    free(dataC); 
    free(dataD); 
    return 0;
}
