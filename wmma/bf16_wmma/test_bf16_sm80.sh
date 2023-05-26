#!/usr/bin/bash

nvcc wmma_bf16_test1.cu -arch sm_80 -o run_wmma_bf16_test1

nvcc wmma_bf16_compare1.cu -arch sm_80 -o run_wmma_bf16_compare1

nvcc wmma_bf16_test2.cu -arch sm_80 -o run_wmma_bf16_test2

nvcc wmma_bf16_compare2.cu -arch sm_80 -o run_wmma_bf16_compare2

nvcc wmma_bf16_compare3.cu -arch sm_80 -o run_wmma_bf16_compare3

nvcc wmma_bf16_compare4.cu -arch sm_80 -o run_wmma_bf16_compare4

nvcc wmma_bf16_compare5.cu -arch sm_80 -o run_wmma_bf16_compare5

nvcc wmma_bf16_test3.cu -arch sm_80 -o run_wmma_bf16_test3

nvcc wmma_bf16_compare3_1.cu -arch sm_80 -o run_wmma_bf16_compare3_1


echo "run_wmma_bf16_test1, reslut: "
./run_wmma_bf16_test1

echo "run_wmma_bf16_compare1, reslut: "
./run_wmma_bf16_compare1

echo "run_wmma_bf16_test2, reslut: "
./run_wmma_bf16_test2

echo "run_wmma_bf16_compare2, reslut: "
./run_wmma_bf16_compare2

echo "run_wmma_bf16_compare3, reslut: "
./run_wmma_bf16_compare3

echo "run_wmma_bf16_compare4, reslut: "
./run_wmma_bf16_compare4

echo "run_wmma_bf16_compare5, reslut: "
./run_wmma_bf16_compare5

echo "run_wmma_bf16_test3, reslut: "
./run_wmma_bf16_test3

echo "run_wmma_bf16_compare3_1, reslut: "
./run_wmma_bf16_compare3_1

rm  run_wmma_bf16_*
