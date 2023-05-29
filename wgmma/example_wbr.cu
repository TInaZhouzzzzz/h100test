#include "cuda.h"
#include "utils.cuh"
#include <iostream>
#include <climits>

// from http://gitlab-software.cambricon.com/wangbingrui/gpu-benchmark/-/blob/master/schedule/warp_schedule_mma.cu  
// from  https://github.com/sunlex0717/gpu-arch-microbenchmark
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

__global__ void warpScheduleKernel(float* input, float* output, uint* clock_start, uint* clock_end){
    int tid = threadIdx.x;
    int laneid = tid & 0x1f;
    int warpid = tid >> 5;

    const int ILPconfig = 6;
    input += tid*6*ILPconfig;
    output += tid*4*ILPconfig;

    float* a = input;
    float* b = input+4*ILPconfig;

    int loop_num = 4*16;
    int inst_num = 6;
    float acc = 0;

    float frag_A[4 * ILPconfig]; // two .f16x2 registers, 8 half elements, 
    float frag_B[2 * ILPconfig];  // one .f16x2 registers, 4 half  elements
    float frag_D[4 * ILPconfig]; //result(fp32) 4 f32 registers

    for(int i = 0;i<4 * ILPconfig;i++){
      frag_A[i] = a[i]; 
    }
    for(int i =0;i<2 * ILPconfig;i++){
      frag_B[i] = b[i]; 
    }
    
    
    for(int i =0;i<4 * ILPconfig;i++){
      frag_D[i] = 0.0f;
    }

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_A[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_B[0]);//?
    float *C = reinterpret_cast<float *>(&frag_D[0]);
    float *D = C;  // D = A*B + D. 


    __syncthreads();
    uint c1 = getClock();
    #pragma unroll
    for (int i = 0; i < loop_num; ++i){
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
            : 
              "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), 
              "r"(B[0]), "r"(B[1]), 
              "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
        );
	if (inst_num >= 2) {
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[4]), "r"(A[5]), "r"(A[6]), "r"(A[7]), 
        "r"(B[2]), "r"(B[3]), 
        "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
    );
	}
	if (inst_num >= 3) {
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[8]), "=f"(D[9]), "=f"(D[10]), "=f"(D[11])
      : "r"(A[8]), "r"(A[9]), "r"(A[10]), "r"(A[11]), 
        "r"(B[4]), "r"(B[5]), 
        "f"(C[8]), "f"(C[9]), "f"(C[10]), "f"(C[11])
    );
	}
	if (inst_num >= 4) {
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[12]), "=f"(D[13]), "=f"(D[14]), "=f"(D[15])
      : "r"(A[12]), "r"(A[13]), "r"(A[14]), "r"(A[15]), 
        "r"(B[6]), "r"(B[7]), 
        "f"(C[12]), "f"(C[13]), "f"(C[14]), "f"(C[15])
    );
	}
	if (inst_num >= 5) {
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[16]), "=f"(D[17]), "=f"(D[18]), "=f"(D[19])
      : "r"(A[16]), "r"(A[17]), "r"(A[18]), "r"(A[19]), 
        "r"(B[8]), "r"(B[9]), 
        "f"(C[16]), "f"(C[17]), "f"(C[18]), "f"(C[19])
    );
	}
	if (inst_num >= 6) {
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[20]), "=f"(D[21]), "=f"(D[22]), "=f"(D[23])
      : "r"(A[20]), "r"(A[21]), "r"(A[22]), "r"(A[23]), 
        "r"(B[10]), "r"(B[11]), 
        "f"(C[20]), "f"(C[21]), "f"(C[22]), "f"(C[23])
    );
	}
        __syncwarp();
    }
    uint c2 = getClock();
    __syncthreads();

    clock_start[tid] = c1;
    clock_end[tid] = c2;

    float* res = output;
    for(int i=0; i < 4*ILPconfig;i++){
      res[i] = frag_D[i]; 
    }
}

uint sumArray(uint* array, int size){
    uint acc = 0;
    for (int i = 0; i < size; ++i){
        acc += array[i];
    }
    return acc;
}


int main(){

    float* input_h;
    float* input_d;
    float* output_h;
    float* output_d;
    uint32_t* clock_start_h;
    uint32_t* clock_end_h;
    uint32_t* clock_start_d;
    uint32_t* clock_end_d;

    int size = 4096*8;

    input_h     = static_cast<float*>(malloc(sizeof(float) * size));
    output_h    = static_cast<float*>(malloc(sizeof(float) * size));
    clock_start_h     = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));
    clock_end_h     = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));


    cudaMalloc(&input_d,  sizeof(float) * size);
    cudaMalloc(&output_d, sizeof(float) * size);
    cudaMalloc(&clock_start_d,  sizeof(uint32_t) * size);
    cudaMalloc(&clock_end_d,  sizeof(uint32_t) * size);

    cudaMemcpy(input_d, input_h, sizeof(float) * size, cudaMemcpyHostToDevice);


/*
 *    const char* cubin_name = "../sass_cubin/warp_schedule.cubin";
 *    const char* kernel_name = "warpSchedule";
 *
 *    printf(">>> SASS Level Warp Scedule Detect\n");
 *    for (int i = 1; i < 8; ++i){
 *        void* kernel_args[4] = {&input_d, &output_d, &clock_end_d, &i};
 *        cudaMemset(clock_end_d, 0, sizeof(uint) * size);
 *        launchSassKernel(cubin_name, kernel_name, gDim, bDim, 0, kernel_args);
 *        cudaMemcpy(clock_end_h, clock_end_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
 *
 *        printf("        Run Warp <0, %d>  Elapsed \t%6u cycle\n", i, sumArray(clock_end_h, 64));
 *        cudaDeviceSynchronize();
 *    }
 *
 */


    int warp_num = 8*4;
    dim3 gDim(1, 1, 1);
    dim3 bDim(warp_num*32, 1, 1);

    printf(">>> CUDA-C Level Warp Schedule Detect\n");
    cudaMemset(clock_start_d, 0, sizeof(uint) * size);
    cudaMemset(clock_end_d, 0, sizeof(uint) * size);
    warpScheduleKernel<<<gDim, bDim>>>(input_d, output_d, clock_start_d, clock_end_d);
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(clock_start_h, clock_start_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(clock_end_h, clock_end_d, sizeof(float) * size, cudaMemcpyDeviceToHost);

    printf("========Run Warp <0..%d>\n", warp_num-1);
    uint32_t clock_base = UINT_MAX;
    for (int j=0; j<=warp_num-1; j+=4) {
        if (clock_base > clock_start_h[j*32]) {
            clock_base = clock_start_h[j*32];
        }
    }
    for (int j=0; j<=warp_num-1; j+=4) {
        printf("warp %d execute from %d to %d\n", j, clock_start_h[j*32]-clock_base, clock_end_h[j*32]-clock_base);
    }
    cudaDeviceSynchronize();

    return 0;
}

