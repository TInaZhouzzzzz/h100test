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

__global__ void test(float* d, float* a, float* b){
  asm volatile(
      "ld.param.u64  %rd1, [_Z4testPfS_S__param_0];\n\t"
      ".reg .b32 f16d<2>;\n\t"
      "wgmma.fence.sync.aligned;\n\t"
      "wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2 {f16d0, f16d1}, %0, %1, 0, 1, 1;\n\t"
      "wgmma.commit_group.sync.aligned;\n\t"
      "wgmma.wait_group.sync.aligned 0;\n\t" 
      "stmatrix.sync.aligned.m8n8.x2.b16 [%0], {f16d0, f16d1};\n\t"
      :"=l"(d): "l"(a),"l"(b)
      );
}

void Initfloat(float * a, const int n) {
  float value;
  for ( int i = 0; i < n; i++ ) {
	value = (float)(rand() % 20 - 10) + (float)(rand() % 20 - 10) / 10.0 + (float)(rand() % 20 - 10) / 100.0 + (float)(rand() % 20 - 10) / 1000.0 + (float)(rand() % 20 - 10) / 10000.0 + (float)(rand() % 20 - 10) / 100000.0 + (float)(rand() % 20 - 10) / 1000000.0 + (float)(rand() % 20 - 10) / 10000000.0 + (float)(rand() % 20 - 10) / 100000000.0;

	a[i] = value;
  }
}

void Initnum(float * a, const int n) {
  FP32 fp32;
  fp32.i = 0x3838;
  for ( int i = 0; i < n; i++ ) {
	  a[i] = fp32.f;
  }
}
void InitZero(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
	  a[i] = 0.0;
  }
}


void show(float * a, const int n) {
  for ( int i=0; i<n; i++){ 
    std::cout << a[i] << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv){
  int size = 512;
  float* host_a=(float*)malloc(sizeof(float) * size);
  float* host_b=(float*)malloc(sizeof(float) * size);
  float* host_d=(float*)malloc(sizeof(float) * size);
  float* device_a=NULL;
  float* device_b=NULL;
  float* device_d=NULL;
  cudaMalloc((void**)(&device_a), sizeof(float) * size);
  cudaMalloc((void**)(&device_b), sizeof(float) * size);
  cudaMalloc((void**)(&device_d), sizeof(float) * size);
  FP32 fp32;
  fp32.i = 0x3838;
  for(int i=0;i<size;i++){
    host_a[i] = fp32.f;
  }
  fp32.i = 0x3c3c;
  for(int i=0;i<size;i++){
    host_b[i] = fp32.f;
  }
  fp32.i = 0x3c000000;
  for(int i=0;i<size;i++){
    host_d[i] = fp32.f;
  }
  cudaMemcpy((void*)device_a, (void*)host_a, sizeof(float)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_b, (void*)host_b, sizeof(float)* size, cudaMemcpyHostToDevice);
  //cudaMemcpy((void*)device_d, (void*)host_d, sizeof(float)* size, cudaMemcpyHostToDevice);

  test<<<4,128>>>(device_d, device_a, device_b);
  cudaDeviceSynchronize();
  cudaMemcpy((void*)host_d, (void*)device_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  fp32.f=host_d[0];
  std::cout<< hex << fp32.i << std::endl;
//show(host_d, size);

}
