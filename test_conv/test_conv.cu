#include <cudnn.h>
#include <iostream>
#include <cstdlib>
using namespace std;
#include "src/helper.h"
#include "cuda_runtime.h"
#include <sys/time.h>
#include <cuda.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda_bf16.h>
 
void InitRand(float * a, const int n) {
  float value;
  for ( int i = 0; i < n; i++ ) {
    value = (float)(rand() % 20 - 10) + (float)(rand() % 20 - 10) / 10.0 + (float)(rand() % 20 - 10) / 100.0 + (float)(rand() % 20 - 10) / 1000.0 + (float)(rand() % 20 - 10) / 10000.0 + (float)(rand() % 20 - 10) / 100000.0 + (float)(rand() % 20 - 10) / 1000000.0 + (float)(rand() % 20 - 10) / 10000000.0 + (float)(rand() % 20 - 10) / 100000000.0;
    a[i] = value;
  }
}
 
void InitZero(float * a, const int n) {
  for ( int i = 0; i < n; i++ ) {
    a[i] = 0.f;
  }
}
 
 
void show(float * a, const int n, const int c, const int h, const int w) {
  for ( int i=0; i<n; i++) {
    for ( int j=0; j<c; j++) {
      for ( int k=0; k<h; k++) {
        for ( int l=0; l<w; l++) {
          // std::cout.width(11);
          std::cout << a[i*c + j*h + k*w + l] << ",";
        }
    std::cout << std::endl;
      }
      std::cout << "next channel" << std::endl;
    }
    std::cout << "next batch" << std::endl;
  }
  std::cout << std::endl;
}
 
int main(int argc, char** argv)
{
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t bias_desc;
 
    //cudnnConvolutionFwdAlgoPerf_t falgo;
    cudnnConvolutionFwdAlgo_t algo;
     
    float *d_input = nullptr;
    float *d_output = nullptr;
    float *d_output2 = nullptr;
    float *d_filter = nullptr;
    float *d_bias = nullptr;
    float *input, *output, *output2, *filter, *bias;
 
    int input_n = 64;
    int input_c = 128;
    int input_h = 16;
    int input_w = 16;
     
    int pad_h = 0;
    int pad_w = 0;
 
    // output size
    int output_n = 16;
    int output_c = 256;
    int output_h = 64;
    int output_w = 64;
 
    // kernel size
    int filter_h = 1;
    int filter_w = 1;
 
    // alpha, beta
    float one = 1.f;
    float zero = 0.f;
 
    cudnnCreate(&cudnn);
 
    /* Create Resources */
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
 
    // Initilziae resources
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_c, input_c, filter_h, filter_w);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    pad_h, pad_w,
                                    1, 1,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);
    cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH);  //zsy
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &output_n, &output_c, &output_h, &output_w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, output_h, output_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, output_c, 1, 1);
     
    int weight_size = output_c * input_c * filter_h * filter_w;
    int bias_size = output_c;
 
    std::cout << "input  size: " << input_n << " " << input_c << " " << input_h << " " << input_w << std::endl;
    std::cout << "output size: " << output_n << " " << output_c << " " << output_h << " " << output_w << std::endl;
    std::cout << "kernel size: " << filter_h << " " << filter_w << std::endl;
    std::cout << "padding size " << pad_h << " " << pad_w << std::endl;
     
    // allocate memory space
    input = (float *)malloc(sizeof(float) * input_n * input_c * input_h * input_w);
    filter = (float *)malloc(sizeof(float) * weight_size);
    output = (float *)malloc(sizeof(float) * output_n * output_c * output_h * output_w); 
    bias = (float *)malloc(sizeof(float) * bias_size);
 
    InitRand(input, input_n * input_c * input_h * input_w);
    input[0] = 12.50390625;
    InitRand(filter, weight_size);
    filter[0] = 1.f;
    InitZero(bias, bias_size);
    InitZero(output, output_n * output_c * output_h * output_w);
 
    std::cout << "Finish init input, filter, bias: " << std::endl;
 
    size_t workspace_size = 0;
    size_t temp_size = 0;
    float *d_workspace = nullptr;
     
    // Algorithm used for convolution
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
 
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &temp_size);
 
    workspace_size = max(workspace_size, temp_size);
    std::cout << "algorithm: " << algo << std::endl;
    std::cout << "workspace size: " << workspace_size << std::endl;
 
 
    cudaMalloc((void**)&d_input,        sizeof(float) * input_n * input_c * input_h * input_w);
    cudaMalloc((void**)&d_filter,       sizeof(float) * weight_size);
    cudaMalloc((void**)&d_output,       sizeof(float) * output_n * output_c * output_h * output_w);
    cudaMalloc((void**)&d_workspace,    sizeof(float) * workspace_size);
    cudaMalloc((void**)&d_bias,         sizeof(float) * bias_size);
     
    cudaMemcpy(d_input, input, sizeof(float) * input_n * input_c * input_h * input_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, sizeof(float) * weight_size, cudaMemcpyHostToDevice);
 
    std::cout << "Finish cudaMemcpy input and filter " << std::endl;
 
    checkCudnnErrors(cudnnConvolutionForward(cudnn, &one, input_desc, d_input, filter_desc, d_filter, conv_desc, algo, d_workspace, workspace_size, &zero, output_desc, d_output));
    cudaMemcpy(output, d_output, sizeof(float) * output_n * output_c * output_h * output_w, cudaMemcpyDeviceToHost);
    std::cout << "Finish compute cudnnConvForward " << std::endl;
 
    //show(output, output_n, output_c, output_h, output_w);
 
    cudaFree(d_input);   
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaFree(d_bias);
 
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
 
    cudaFree(d_input);   
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_workspace);
    cudaFree(d_bias);
 
    cudnnDestroy(cudnn);
}
