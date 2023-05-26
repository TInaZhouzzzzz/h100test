#!/usr/bin/bash

nvcc wmma_fp16_test1.cu -arch sm_90 -o run_wmma_fp16_test1

nvcc wmma_fp16_compare1_1.cu -arch sm_90 -o run_wmma_fp16_compare1_1

nvcc wmma_fp16_compare1_2.cu -arch sm_90 -o run_wmma_fp16_compare1_2

nvcc wmma_fp16_compare1_3.cu -arch sm_90 -o run_wmma_fp16_compare1_3

nvcc wmma_fp16_compare1_4.cu -arch sm_90 -o run_wmma_fp16_compare1_4

nvcc wmma_fp16_compare1_5.cu -arch sm_90 -o run_wmma_fp16_compare1_5

nvcc wmma_fp16_test2.cu -arch sm_90 -o run_wmma_fp16_test2

nvcc wmma_fp16_compare2_1.cu -arch sm_90 -o run_wmma_fp16_compare2_1

nvcc wmma_fp16_compare2_2.cu -arch sm_90 -o run_wmma_fp16_compare2_2

nvcc wmma_fp16_compare2_3.cu -arch sm_90 -o run_wmma_fp16_compare2_3

nvcc wmma_fp16_compare2_4.cu -arch sm_90 -o run_wmma_fp16_compare2_4

nvcc wmma_fp16_test3.cu -arch sm_90 -o run_wmma_fp16_test3

nvcc wmma_fp16_compare3_1.cu -arch sm_90 -o run_wmma_fp16_compare3_1

nvcc wmma_fp16_compare3_2.cu -arch sm_90 -o run_wmma_fp16_compare3_2

nvcc wmma_fp16_compare3_3.cu -arch sm_90 -o run_wmma_fp16_compare3_3

nvcc wmma_fp16_compare3_4.cu -arch sm_90 -o run_wmma_fp16_compare3_4

nvcc wmma_fp16_compare3_5.cu -arch sm_90 -o run_wmma_fp16_compare3_5

nvcc wmma_fp16_compare3_6.cu -arch sm_90 -o run_wmma_fp16_compare3_6



echo "run_wmma_fp16_test1, reslut: "
./run_wmma_fp16_test1

echo "run_wmma_fp16_compare1_1, reslut: "
./run_wmma_fp16_compare1_1

echo "run_wmma_fp16_compare1_2, reslut: "
./run_wmma_fp16_compare1_2

echo "run_wmma_fp16_compare1_3, reslut: "
./run_wmma_fp16_compare1_3

echo "run_wmma_fp16_compare1_4, reslut: "
./run_wmma_fp16_compare1_4

echo "run_wmma_fp16_compare1_5, reslut: "
./run_wmma_fp16_compare1_5


echo "run_wmma_fp16_test2, reslut: "
./run_wmma_fp16_test2

echo "run_wmma_fp16_compare2_1, reslut: "
./run_wmma_fp16_compare2_1

echo "run_wmma_fp16_compare2_2, reslut: "
./run_wmma_fp16_compare2_2

echo "run_wmma_fp16_compare2_3, reslut: "
./run_wmma_fp16_compare2_3

echo "run_wmma_fp16_compare2_4, reslut: "
./run_wmma_fp16_compare2_4


echo "run_wmma_fp16_test3, reslut: "
./run_wmma_fp16_test3

echo "run_wmma_fp16_compare3_1, reslut: "
./run_wmma_fp16_compare3_1

echo "run_wmma_fp16_compare3_2, reslut: "
./run_wmma_fp16_compare3_2

echo "run_wmma_fp16_compare3_3, reslut: "
./run_wmma_fp16_compare3_3

echo "run_wmma_fp16_compare3_4, reslut: "
./run_wmma_fp16_compare3_4

echo "run_wmma_fp16_compare3_5, reslut: "
./run_wmma_fp16_compare3_5

echo "run_wmma_fp16_compare3_6, reslut: "
./run_wmma_fp16_compare3_6

rm  run_wmma_fp16_*
