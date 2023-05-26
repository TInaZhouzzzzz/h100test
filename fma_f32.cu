#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include <iomanip>      // std::setprecision

#include<cuda_fp16.h>
#include "include/half.hpp"
using namespace std;
using half_float::half;
union FP32{
    float f;
    unsigned int i;
};

__global__ void test(float* A, float* B, float* C, float* D){
    asm("fma.rn.f32  %0, %1, %2, %3;" : "=f"(D[0]) : "f"(A[0]) , "f"(B[0]) , "f"(C[0]));
}

int main(int argc, char** argv){
    int size = 1;
    float *dataA = (float*)malloc(sizeof(float) * size);
    float *dataB = (float*)malloc(sizeof(float) * size);
    float *dataC = (float*)malloc(sizeof(float) * size);
    float *dataD = (float*)malloc(sizeof(float) * size);
    float *d_dataA = NULL;
    float *d_dataB = NULL;
    float *d_dataC = NULL;
    float *d_dataD = NULL;
    cudaMalloc((void**)&d_dataA, sizeof(float) * size);
    cudaMalloc((void**)&d_dataB, sizeof(float) * size);
    cudaMalloc((void**)&d_dataC, sizeof(float) * size);
    cudaMalloc((void**)&d_dataD, sizeof(float) * size);
    FP32 fp32;

    fp32.i = 0x5d840000;    dataA[size-1] = fp32.f;
    fp32.i = 0xa2300000;    dataB[size-1] = fp32.f;
    fp32.i = 0x01000000;    dataC[size-1] = fp32.f;

//  show(dataA, size);


    for(int i=0;i<size;i++){
        dataD[i] = 0;
    }
    cudaMemcpy(d_dataA,dataA,sizeof(float) * size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB,dataB,sizeof(float) * size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataC,dataC,sizeof(float) * size,cudaMemcpyHostToDevice);
    test<<<1, 1>>> (d_dataA, d_dataB, d_dataC, d_dataD);
    cudaMemcpy(dataD,d_dataD,sizeof(float) * size, cudaMemcpyDeviceToHost);

    fp32.f = dataD[size-1];
    std::cout <<std::hex << fp32.i << std::endl;

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
