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
  float d_array[4];
  for (int i = 0; i < 4; ++i) {
    d_array[i] = d[i];
  }

  asm volatile(
      ".reg .b32 ra<4>, rd<4>;\n\t"
      "wgmma.fence.sync.aligned;\n\t"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e5m2 {%0, %1, %2, %3}, %4, %5, 0, -1, -1;\n\t"
      "wgmma.commit_group.sync.aligned;\n\t"
      "wgmma.wait_group.sync.aligned 0;\n\t" 
      : "+f"(d_array[0]), "+f"(d_array[1]), "+f"(d_array[2]), "+f"(d_array[3])
      : "l"(a),"l"(b)
      );
  for (int i = 0; i < 4; ++i) {
    d[i] = d_array[i];
  }
}

void InitOne(float * a, const int n) {
  FP32 fp32;
  fp32.i = 0x38383838;
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
  for(int i=0;i<size;i++){
    host_a[i] =0.0;
    host_d[i] = 0.0;
  }
  InitOne(host_b, size);

  FP32 fp32;
  fp32.i = 0x70400000;  host_a[0]=fp32.f;

  cudaMemcpy((void*)device_a, (void*)host_a, sizeof(float)* size, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)device_b, (void*)host_b, sizeof(float)* size, cudaMemcpyHostToDevice);

  test<<<4,128>>>(device_d, device_a, device_b);
  cudaDeviceSynchronize();
  cudaMemcpy((void*)host_d, (void*)device_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  fp32.f=host_d[0];
  std::cout<< hex << fp32.i << std::endl;
//show(host_d, size);

}

