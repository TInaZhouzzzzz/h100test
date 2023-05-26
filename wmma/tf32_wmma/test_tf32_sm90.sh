#!/usr/bin/bash

nvcc wmma_tf32_test1.cu -arch sm_90 -o run_wmma_tf32_test1

nvcc wmma_tf32_compare1.cu -arch sm_90 -o run_wmma_tf32_compare1

nvcc wmma_tf32_test2.cu -arch sm_90 -o run_wmma_tf32_test2

nvcc wmma_tf32_compare2.cu -arch sm_90 -o run_wmma_tf32_compare2

nvcc wmma_tf32_test3.cu -arch sm_90 -o run_wmma_tf32_test3

nvcc wmma_tf32_test4.cu -arch sm_90 -o run_wmma_tf32_test4

nvcc wmma_tf32_test5.cu -arch sm_90 -o run_wmma_tf32_test5


echo "run_wmma_tf32_test1, reslut: "
./run_wmma_tf32_test1

echo "run_wmma_tf32_compare1, reslut: "
./run_wmma_tf32_compare1

echo "run_wmma_tf32_test2, reslut: "
./run_wmma_tf32_test2

echo "run_wmma_tf32_compare2, reslut: "
./run_wmma_tf32_compare2

echo "run_wmma_tf32_test3, reslut: "
./run_wmma_tf32_test3

echo "run_wmma_tf32_test4, reslut: "
./run_wmma_tf32_test4

echo "run_wmma_tf32_test5, reslut: "
./run_wmma_tf32_test5

rm  run_wmma_tf32_*
