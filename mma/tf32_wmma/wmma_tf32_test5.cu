#include <iostream>
#include <cstdlib>
using namespace std;

#include <sys/time.h>
#include <cuda.h>
#include <unistd.h>
#include <stdlib.h>


union FP32
{
    unsigned int i;
    float f;
};

__global__ void test(float* d, float* a, float* b, float* c){
  asm volatile(
    "ld.param.u64    %rd1, [_Z4testPfS_S_S__param_0];\n\t"
    ".reg .b32 a<4>, b<4>, c<8>,d<8>;\n\t"
    "wmma.load.a.sync.aligned.m16n16k8.global.row.tf32 {a0, a1, a2, a3}, [%1];\n\t"
    "wmma.load.b.sync.aligned.m16n16k8.global.col.tf32 {b0, b1, b2, b3}, [%2];\n\t"
    "wmma.load.c.sync.aligned.m16n16k8.global.row.f32 {c0, c1, c2, c3, c4, c5, c6, c7}, [%3];\n\t"
    "wmma.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 {d0,d1,d2,d3,d4,d5,d6,d7},  {a0, a1, a2, a3}, {b0, b1, b2, b3}, {c0, c1, c2, c3, c4, c5, c6, c7};\n\t"
    "wmma.store.d.sync.aligned.m16n16k8.global.row.f32 [%0], {d0,d1,d2,d3,d4,d5,d6,d7};" : "=l"(d): "l"(a), "l"(b), "l"(c));
}

void InitZero(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 0.0;
  }
}

void InitOne(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
      a[i] = 1.0;
  }
}

void Init(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
      a[i] = 1.0 * float(i);
  }
}

void Init_3f800000(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
      FP32 fp32;
      fp32.i = 0x3f800000;
      a[i] = fp32.f;
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
  float* host_a=(float*)malloc(sizeof(float) * size/2);
  float* host_b=(float*)malloc(sizeof(float) * size/2);
  float* host_c=(float*)malloc(sizeof(float) * size);
  float* host_d=(float*)malloc(sizeof(float) * size);
  float* device_a=NULL;
  float* device_b=NULL;
  float* device_c=NULL;
  float* device_d=NULL;
  cudaMalloc((void**)(&device_a), sizeof(float) * size/2);
  cudaMalloc((void**)(&device_b), sizeof(float) * size/2);
  cudaMalloc((void**)(&device_c), sizeof(float) * size);
  cudaMalloc((void**)(&device_d), sizeof(float) * size);
  InitZero(host_a, size/2);
  Init_3f800000(host_b, size/2);
  InitZero(host_c, size);
  InitZero(host_d, size);
  FP32 fp32;
  fp32.i = 0x3f800000; host_a[0]=fp32.f;
  fp32.i = 0x3f800000; host_a[1]=fp32.f;
  fp32.i = 0x3f800000; host_a[4]=fp32.f;
  fp32.i = 0x3f800000; host_a[5]=fp32.f;

  fp32.i = 0x4c000000; host_c[0]=fp32.f;

  cudaMemcpy((void*)device_a, (void*)host_a, sizeof(float)* size/2, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_b, (void*)host_b, sizeof(float)* size/2, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_c, (void*)host_c, sizeof(float)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_d, (void*)host_d, sizeof(float)* size, cudaMemcpyHostToDevice);

  test<<<1,32>>>(device_d, device_a, device_b, device_c);
  cudaDeviceSynchronize();

  cudaMemcpy((void*)host_d, (void*)device_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  fp32.f=host_d[0];
//std::cout<< host_d[0] << std::endl;
  std::cout<< hex << fp32.i << std::endl;
//show(host_d, size);
}
