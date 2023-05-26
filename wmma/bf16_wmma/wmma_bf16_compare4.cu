#include <iostream>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>
using namespace std;

#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstdlib>


union FP32
{
    unsigned int i;
    float f;
};


union BF16
{
    unsigned short int i;
    __nv_bfloat16 f;
};

__global__ void test(float* dst, __nv_bfloat16* a, __nv_bfloat16* b, float* c){
  asm volatile(
//  "ld.param.u64    %rd1, [_Z4testPfP6__halfS1_S__param_0];\n\t"
    "ld.param.u64 	%rd1, [_Z4testPfP13__nv_bfloat16S1_S__param_0];\n\t"
    ".reg .b32 a<4>, b<4>, c<8>,d<8>;\n\t"
    "wmma.load.a.sync.aligned.m16n16k16.global.row.bf16 {a0, a1, a2, a3}, [%1];\n\t"
    "wmma.load.b.sync.aligned.m16n16k16.global.col.bf16 {b0, b1, b2, b3}, [%2];\n\t"
    "wmma.load.c.sync.aligned.m16n16k16.global.row.f32 {c0, c1, c2, c3, c4, c5, c6, c7}, [%3];\n\t"
    "wmma.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 {d0,d1,d2,d3,d4,d5,d6,d7},  {a0, a1, a2, a3}, {b0, b1, b2, b3}, {c0, c1, c2, c3, c4, c5, c6, c7};\n\t"
    "wmma.store.d.sync.aligned.m16n16k16.global.row.f32 [%0], {d0,d1,d2,d3,d4,d5,d6,d7};" : "=l"(dst): "l"(a), "l"(b), "l"(c));
}

void InitOne(__nv_bfloat16* a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 1.0;
  }
}

void InitZero(__nv_bfloat16* a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 0.0;
  }
}


void InitZero_float(float* a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 0.0;
  }
}

void show(float * a, const int n) {
  std::cout << std::endl;
  for ( int i=0; i<n; i++){ 
    std::cout<<a[i] << std::endl;
  }
  std::cout << std::endl;
}


int main(int argc, char** argv){
  int size = 256;
  __nv_bfloat16* host_a=(__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * size);
  __nv_bfloat16* host_b=(__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * size);
  float* host_c=(float*)malloc(sizeof(float) * size);
  float* host_d=(float*)malloc(sizeof(float) * size);
  __nv_bfloat16* device_a=NULL;
  __nv_bfloat16* device_b=NULL;
  float* device_c=NULL;
  float* device_d=NULL;
  cudaMalloc((void**)(&device_a), sizeof(__nv_bfloat16) * size);
  cudaMalloc((void**)(&device_b), sizeof(__nv_bfloat16) * size);
  cudaMalloc((void**)(&device_c), sizeof(float) * size);
  cudaMalloc((void**)(&device_d), sizeof(float) * size);
  InitZero(host_a, size);
  InitOne(host_b, size);
  InitZero_float(host_c, size);
  InitZero_float(host_d, size);

  BF16 bf16;
  bf16.i = 0x3f80; host_a[0]=bf16.f;
  bf16.i = 0x3f80; host_a[1]=bf16.f;
  bf16.i = 0x3f80; host_a[2]=bf16.f;
  bf16.i = 0x3f80; host_a[3]=bf16.f;
  bf16.i = 0x3f80; host_a[4]=bf16.f;
  bf16.i = 0x3f80; host_a[5]=bf16.f;
  bf16.i = 0x3f80; host_a[6]=bf16.f;
  bf16.i = 0x3f80; host_a[7]=bf16.f;


  FP32 fp32;
  fp32.i = 0x4c000000; host_c[0]=fp32.f;

  cudaMemcpy((void*)device_a, (void*)host_a, sizeof(__nv_bfloat16)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_b, (void*)host_b, sizeof(__nv_bfloat16)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_c, (void*)host_c, sizeof(float)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_d, (void*)host_d, sizeof(float)* size, cudaMemcpyHostToDevice);

  test<<<1,32>>>(device_d, device_a, device_b, device_c);
  cudaDeviceSynchronize();

  cudaMemcpy((void*)host_d, (void*)device_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

//FP32 fp32;
  fp32.f=host_d[0];
//std::cout<< host_d[0] << std::endl;
  std::cout<< hex << fp32.i << std::endl;
//show(host_d, size);
}
