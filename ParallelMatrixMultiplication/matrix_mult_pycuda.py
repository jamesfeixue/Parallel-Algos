#!/usr/bin/env python

"""
.
.
.
Python Code
.
.
.
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


#%%
from pycuda import driver, compiler, gpuarray, tools
import time
#%%
import pycuda.autoinit

class Transpose:
    def transpose(self, a_cpu):
        print("--"*40)
        print("transpose checks")
        print("--"*40)

        a_cpu = a_cpu.astype(np.float32)
#        print(a_cpu)
        block_size = 16
        height = np.int32(a_cpu.shape[0])
        width = np.int32(a_cpu.shape[1])

        
        output_gpu = gpuarray.empty((width, height), np.float32)

        
        transpose_kernel = """
        #include <stdio.h>
        __global__ void Transpose(float *input, float *output, int input_width, int input_height){
            int tx = blockIdx.x*blockDim.x + threadIdx.x; 
            int ty = blockIdx.y*blockDim.y + threadIdx.y;
            int Y = input_height;
            int X = input_width; 
            // printf("tx:%d, ty:%d, Y:%d, X:%d, input[tx*X+ty]:%d \\n", tx, ty, Y, X, input[tx*X+ty]);            
            if (tx<Y && ty<X){
                    output[ty*Y+tx] = input[tx*X+ty];  
            }
        }
        
        """
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory
        input_gpu = gpuarray.to_gpu(a_cpu)
         
        # get the kernel code from the template
        # by specifying the constant MATRIX_SIZE
        kernel_code = transpose_kernel 
        mod = compiler.SourceModule(kernel_code)
        transpose = mod.get_function("Transpose")
        grid_x = int(np.ceil(height/float(block_size)))
        
        grid_y = int(np.ceil(width/float(block_size)))
        
#        print "grid_x", grid_x
#        print "gird_x.type", type(grid_x)
         
        # call the kernel on the card
        start = time.time()
        transpose(
                input_gpu, 
                output_gpu, 
                width, height, 
                block = (block_size, block_size, 1), grid = (grid_x, grid_y, 1))#line blocksize
        kernel_time = time.time() - start
        
        result = output_gpu.get()
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time


#%%
#generic transpose algo    
def matrix_transpose(matrix): 
    start = time.time() 
    matrix = np.matrix(matrix)
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    new_matrix = np.empty((columns, rows))
    new_matrix = np.matrix(new_matrix)

    for i in range(0, rows): 
        for j in range(0, columns): 
            new_matrix[j,i] = matrix[i, j]
    
    running_time = time.time() - start
            
    return new_matrix, running_time

#%%
#M = [[1, 2, 3], [1, 2, 3]]
##matrix_transpose(M)
##j = [1, 2, 3]
##np.matrix(j).shape
##matrix_transpose(j)
#    
#kernel_transpose = Transpose() 
#result, kernel_time, total_time, check = kernel_transpose.transpose(M)
#print(result, kernel_time, total_time, check)
    
"""
Calculate the transpose of them using 3 transpose algorithms (2 parallel, 1 serial) respectively. 
Record the running time of each call for each of the algorithm.
"""
 
def transpose_check(): 
    M = 2
    N = 3
    kernel_times = []
    total_times = []
    serial_times = []
    multiple = []
    
    for i in range(1, 100): 
        rows = M*i
        columns = N*i
        matrix = np.random.rand(rows, columns)
        
        kernel_transpose = Transpose() 
        result, kernel_time, total_time = kernel_transpose.transpose(matrix)
        kernel_times.append(kernel_time)
        total_times.append(total_time)
        
        transposed = np.transpose(matrix).astype(np.float32)
        
        result_2, serial_time = matrix_transpose(matrix)
        serial_times.append(serial_time)
        
#        print(transposed)
#        print("-" * 80)
#        print result
        multiple.append(i)
    
    """Plotting"""
    plt.plot(multiple, serial_times, color='b', label="serial")
    plt.plot(multiple, kernel_times, color='g', label="kernel")
    plt.plot(multiple, total_times, color='r', label="kernel+load") 
    plt.legend(loc='upper left')
    plt.title('CUDA Transpose Times')
    plt.ylabel('Time')
    plt.xlabel('Iteration')
    plt.savefig('cuda_transpose.png')
    
    print (transposed==result)
    return (transposed==result)
    
#transpose_check() 
    

    
#%%
"""


MATRIX MULTIPLICATION




"""
#%%
import pycuda.autoinit

class MatrixMultiply:
    
    def matrix_mul_naive(self, a_cpu):

        print("-"*80)
        print("naive")
        print("-"*80)
        
        a_cpu = a_cpu.astype(np.float32)
        
        block_size = 32
        height = np.int32(a_cpu.shape[0])
        width = np.int32(a_cpu.shape[1])
        
        output_gpu = gpuarray.empty((height, height), np.float32)
        grid_x = int(np.ceil(height/float(block_size)))
        grid_y = int(np.ceil(height/float(block_size)))

        matrix_multiplication_kernel = """
        #include <stdio.h>
        __global__ void MatMul(float *a_gpu, float *output, int input_width, int input_height){
        
            int ty = blockIdx.y*blockDim.y + threadIdx.y;
            int tx = blockIdx.x*blockDim.x + threadIdx.x; 
            
            int Y = input_height;
            int X = input_width; 
            
            
            if( (ty < Y)  && (tx < Y) ) {
                float summation = 0; 
                for (int k = 0; k < X; k++){
                        float a_value = a_gpu[ty * X + k]; 
                        float b_value = a_gpu[k + X * tx]; 
                        summation += a_value * b_value; 
                        // printf("tx:%d, ty:%d, k:%d, temp:%d, a_value:%d, b_value:%d\\n", tx, ty, k, a_value, b_value);
                    }
                output[ty* Y + tx] = summation;
                 
                }
            __syncthreads();

            }
        
        """
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(a_cpu)
         
        kernel_code = matrix_multiplication_kernel 
        mod = compiler.SourceModule(kernel_code)
        MatrixMultiplication = mod.get_function("MatMul")
         
        # call the kernel on the card
        start = time.time()
        
        MatrixMultiplication(
                a_gpu, 
                output_gpu, 
                width, height, 
                block = (block_size, block_size, 1), grid = (grid_x, grid_y, 1))
        #line blocksize
        kernel_time = time.time() - start
        
        result = output_gpu.get()
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time
    
    
    #%%
    
    def matrix_mul_optimized1(self, a_cpu):

        print("-"*80)
        print("opt1")
        print("-"*80)
        
        a_cpu = a_cpu.astype(np.float32)
        
        block_size = 32
        height = np.int32(a_cpu.shape[0])
        width = np.int32(a_cpu.shape[1])
        
        output_gpu = gpuarray.empty((height, height), np.float32)
        grid_x = int(np.ceil(height/float(block_size)))
        grid_y = int(np.ceil(height/float(block_size)))
        grid_size = np.int32(grid_x)
        
        matrix_multiplication_kernel = """
            #include <stdio.h>
            #include <math.h>
            #define tile_size 32
            
            __global__ void MatMul(float *a_gpu, float *output, const int input_height, const int input_width, const int grid_x){
            
            int ty = blockIdx.y*blockDim.y + threadIdx.y;
            int tx = blockIdx.x*blockDim.x + threadIdx.x; 
            
            int Y = input_height;
            int X = input_width; 
   
            __shared__ float A_shared[tile_size][tile_size];  

   
            float summation = 0;   

            for(int i=0; i < grid_x ; i++)
                {
                 if((i * tile_size + threadIdx.x < X) && (ty < Y)) {
                    A_shared[threadIdx.y][threadIdx.x] = a_gpu[ty * X + i * tile_size + threadIdx.x];
                    }
                 else {
                    A_shared[threadIdx.y][threadIdx.x] = 0;
                    }

                __syncthreads();

                for(int j = 0; j < tile_size; j++)
                    {
                    summation += A_shared[threadIdx.y][j] * a_gpu[j + i * tile_size + tx * X]; 
                    }
                __syncthreads();
                }
            __syncthreads();


            if((ty < Y) && (tx < Y))
                {
                output[ty * Y + tx] = summation;
                }
            __syncthreads();
            }         
            

        """
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(a_cpu)
         
        kernel_code = matrix_multiplication_kernel 
        mod = compiler.SourceModule(kernel_code)
        MatrixMultiplication = mod.get_function("MatMul")
         
        # call the kernel on the card
        start = time.time()
        
        MatrixMultiplication(
                a_gpu, 
                output_gpu, 
                height, width, 
                grid_size, 
                block = (block_size, block_size, 1), grid = (grid_x, grid_y, 1))
        #line blocksize
        kernel_time = time.time() - start
        
        result = output_gpu.get()
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time

#%%

    def matrix_mul_optimized2(self, a_cpu):

        print("-"*80)
        print("opt2")
        print("-"*80)
        
        a_cpu = a_cpu.astype(np.float32)
        
        block_size = 32
        height = np.int32(a_cpu.shape[0])
        width = np.int32(a_cpu.shape[1])
        
        output_gpu = gpuarray.empty((height, height), np.float32)
        grid_x = int(np.ceil(height/float(block_size)))
        grid_y = int(np.ceil(height/float(block_size)))
        grid_size = np.int32(grid_x)
        
        matrix_multiplication_kernel = """
            #include <stdio.h>
            #include <math.h>
            #define tile_size 32
            
            __global__ void MatMul(float *a_gpu, float *output, const int input_height, const int input_width, const int grid_x){
            
            int ty = blockIdx.y*blockDim.y + threadIdx.y;
            int tx = blockIdx.x*blockDim.x + threadIdx.x; 
            
            const int Y = input_height;
            const int X = input_width; 
   
            __shared__ float A_shared[tile_size][tile_size];  
            __shared__ float B_shared[tile_size][tile_size]; 

            int threadx = threadIdx.x; 
            int thready = threadIdx.y; 
            
            float summation = 0;   

            for(int i=0; i < grid_x ; i++)
                {
                 if((i * tile_size + threadx < X) && (ty < Y)) {
                    A_shared[thready][threadx] = a_gpu[ty * X + i * tile_size + threadx];
                    }
                 else {
                    A_shared[thready][threadx] = 0;
                    }
                 
                 if((i * tile_size + thready < X) && (tx < Y)){
                    B_shared[thready][threadx] = a_gpu[tx * X + i * tile_size + thready];
                    }
                 else
                    {
                    B_shared[ty][tx] = 0;
                    }

                __syncthreads();

                for(int j = 0; j < tile_size; j++)
                    {
                    summation += A_shared[thready][j] * B_shared[j][threadx]; 
                    }
                __syncthreads();
                }
            __syncthreads();


            if((ty < Y) && (tx < Y))
                {
                output[ty * Y + tx] = summation;
                }
            __syncthreads();
            }         
            

        """
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(a_cpu)
         
        kernel_code = matrix_multiplication_kernel 
        mod = compiler.SourceModule(kernel_code)
        MatrixMultiplication = mod.get_function("MatMul")
         
        # call the kernel on the card
        start = time.time()
        
        MatrixMultiplication(
                a_gpu, 
                output_gpu, 
                height, width, 
                grid_size, 
                block = (block_size, block_size, 1), grid = (grid_x, grid_y, 1))
        #line blocksize
        kernel_time = time.time() - start
        
        result = output_gpu.get()
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time


        
#%%

def check_matmul(): 
    M = 2
    N = 3
    naive_kernel_times = []
    naive_total_times = []
    opt1_kernel_times = []
    opt1_total_times = []
    opt2_kernel_times = []
    opt2_total_times = []
    
    iteration = []
    
    for i in range(1, 10): 
        rows = M*i
        columns = N*i
        matrix = np.random.rand(rows, columns)
#        matrix = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        matrix = matrix.astype(np.float32)
        transpose = np.transpose(matrix)
        transpose = transpose.astype(np.float32)
        
        kernel_multiply = MatrixMultiply() 
        
        naive_result, naive_kernel_time, naive_total_time = kernel_multiply.matrix_mul_naive(matrix)
        naive_kernel_times.append(naive_kernel_time)
        naive_total_times.append(naive_total_time)
        
        opt1_result, opt1_kernel_time, opt1_total_time = kernel_multiply.matrix_mul_optimized1(matrix)
        opt1_kernel_times.append(opt1_kernel_time)
        opt1_total_times.append(opt1_total_time)
#        
        opt2_result, opt2_kernel_time, opt2_total_time = kernel_multiply.matrix_mul_optimized2(matrix)
        opt2_kernel_times.append(opt2_kernel_time)
        opt2_total_times.append(opt2_total_time)
        
        cpu_multiply = np.matmul(matrix, transpose)
        cpu_multiply = cpu_multiply.astype(np.float32)
      
        iteration.append(i)
    
#    """Plotting"""
    plt.plot(iteration, naive_kernel_times, color='b', label="naive_kernel")
    plt.plot(iteration, naive_total_times, color='b', linestyle = '--', label="naive_total")
    plt.plot(iteration, opt1_kernel_times, color='r', label="opt1_kernel")
    plt.plot(iteration, opt1_total_times, color='r', linestyle = '--', label="opt1_total")
    plt.plot(iteration, opt2_kernel_times, color='g', label="opt2_kernel")
    plt.plot(iteration, opt2_total_times, color='g', linestyle = '--', label="opt2_total")
    plt.legend(loc='upper left')
    plt.title('CUDA Mat Mul Times')
    plt.ylabel('Time')
    plt.xlabel('Iteration')
    plt.savefig('cuda_matmul.png')
    
    print("-"*80)
    print("Matrix Multiplication Checks")
    print("-"*80)
    print (cpu_multiply)
    print("-"*80)
    print (naive_result)
    print("-"*80)
    print(opt1_result)
    print("-"*80)
    print(opt2_result)
    print("-"*80)
    print (opt1_result==naive_result)
    print("*"*10)
    print(opt1_result == opt2_result)
    print("-"*80)
    print("complete")
    return (cpu_multiply==naive_result)

check_matmul()