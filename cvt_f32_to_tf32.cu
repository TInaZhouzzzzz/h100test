#include<cuda.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include <iomanip>      // std::setprecision

using namespace std;
union FP32{
    float f;
    unsigned int i;
};

__global__ void test_cvt_f32_to_f16(float* A, short int* C, int N){
    for(int i=0;i<N;i++){
        //asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n\t":"=h"(C[i]):"f"(A[i]),"f"(B[i]));
        asm("cvt.rn.relu.satfinite.f16.f32 %0, %1;\n\t":"=h"(C[i]):"f"(A[i]));
    }
}

__global__ void test_cvt_f32_to_tf32(float* A, unsigned int* C, int N){
    for(int i=0;i<N;i++){
        //asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n\t":"=h"(C[i]):"f"(A[i]),"f"(B[i]));
        asm("cvt.rna.satfinite.tf32.f32 %0, %1;\n\t":"=r"(C[i]):"f"(A[i]));
    }
}

void Initfloat(float * a, const int n) {
  float value;
  for ( int i = 0; i < n; i++ ) {
    value = (float)(rand() % 20 - 10) + (float)(rand() % 20 - 10) / 10.0 + (float)(rand() % 20 - 10) / 100.0 + (float)(rand() % 20 - 10) / 1000.0 + (float)(rand() % 20 - 10) / 10000.0 + (float)(rand() % 20 - 10) / 100000.0 + (float)(rand() % 20 - 10) / 1000000.0 + (float)(rand() % 20 - 10) / 10000000.0 + (float)(rand() % 20 - 10) / 100000000.0;

    a[i] = value;
  }
}

void InitZero(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
      a[i] = 0.0;
  }
}


void show(float * a, const int n) {
  std::cout << std::endl;
  for ( int i=0; i<n; i++){ 
    std::cout<< std::setprecision(20) << a[i] << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv){
    int size = 10;
    float *dataA = (float*)malloc(sizeof(float) * size);
    unsigned int *dataC = (unsigned int*)malloc(sizeof(unsigned int) * size);
    float *d_dataA = NULL;
    unsigned  int *d_dataC = NULL;
    cudaMalloc((void**)&d_dataA, sizeof(float) * size);
    cudaMalloc((void**)&d_dataC, sizeof(unsigned int) * size);
    FP32 fp32;

    Initfloat(dataA, size);
    /* Nan */
    fp32.i = 0x7fffffff;    dataA[size-1] = fp32.f;
    fp32.i = 0xffffffff;    dataA[size-2] = fp32.f;

    /* inf */
    fp32.i = 0x7f800000;    dataA[size-3] = fp32.f;
    fp32.i = 0xff800000;    dataA[size-4] = fp32.f;

    /* 0 */
    fp32.i = 0x00000000;    dataA[size-5] = fp32.f;
    fp32.i = 0x80000000;    dataA[size-6] = fp32.f;

    /* overflow */
    fp32.i = 0x7f7ffeba;    dataA[size-7] = fp32.f;
    fp32.i = 0xff7ffeba;    dataA[size-8] = fp32.f;

    /* random normal */
    fp32.i = 0x43acad91;    dataA[size-9] = fp32.f;
    fp32.i = 0xc3acad91;    dataA[size-10] = fp32.f;


//  /* underflow */
//  fp32.i = 0x36801000;    dataA[count] = fp32.f;

//  show(dataA, size);


    for(int i=0;i<size;i++){
        dataC[i] = 0;
    }
    cudaMemcpy(d_dataA,dataA,sizeof(float) * size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataC,dataC,sizeof(unsigned int) * size,cudaMemcpyHostToDevice);
    test_cvt_f32_to_tf32<<<1, 1>>> (d_dataA, d_dataC, size);
    cudaMemcpy(dataC,d_dataC,sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);
    std::cout << std::endl;
    
    for(int i=0;i<size;i++){
        FP32 fp32_i;
        FP32 tf32_o;
        fp32_i.f = dataA[i];
        tf32_o.i = dataC[i];
        cout<< "f32 / tf32 hex format:,"<<hex<<fp32_i.i<<"," <<tf32_o.i <<endl;
        cout<< "f32 / tf32 dec format:,"<< std::setprecision(20) <<fp32_i.f<<"," <<tf32_o.f<<endl;
        std::cout << std::endl;
    }

    cudaFree(d_dataA);
    cudaFree(d_dataC);
    free(dataA);
    free(dataC); 
    return 0;
}
