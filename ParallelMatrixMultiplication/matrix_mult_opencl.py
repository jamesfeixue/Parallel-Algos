
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import time

import pyopencl as cl
import pyopencl.array
import numpy as np

#%%

class Transpose:
    def transpose(self, a_cpu):
        print("--"*40)
        print("transpose checks")
        print("--"*40)
        
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # Set up a command queue:
        ctx = cl.Context(devs)
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        a = np.mat(a_cpu).astype(np.float32)

        height = np.array((a_cpu.shape[0])).astype(np.int32)
        width = np.array((a_cpu.shape[1])).astype(np.int32)
        
        c = np.zeros((width, height), np.float32) 
        work_group_size = (np.int32(16), np.int32(16))
        NDRange_size = (np.int32(np.ceil(height / 16.0)*16), np.int32(np.ceil(width / 16.0)*16))
        
        transpose_kernel = """
        __kernel void matrix_transpose(__global float* a, __global float* c, __global int* input_height, __global int* input_width){

            int tx = get_global_id(1); 
            int ty = get_global_id(0);
            int Y = input_height[0];
            int X = input_width[0]; 
           
            if ((tx < X) && (ty < Y)) {
                    c[tx * Y + ty] = a[ty * X + tx]; 
            } 
        }
        
        """
        
        prg = cl.Program(ctx, transpose_kernel).build()
        func = prg.matrix_transpose
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory

        a_gpu = cl.array.to_device(queue, a)
        c_gpu = cl.array.to_device(queue, c)
        height_gpu = cl.array.to_device(queue, height)
        width_gpu = cl.array.to_device(queue, width)
        
        event = func(queue, NDRange_size, work_group_size, a_gpu.data, c_gpu.data, height_gpu.data, width_gpu.data)
        event.wait()
        result = c_gpu.get() 
        
        kernel_time = (event.profile.end - event.profile.start)/(1e10) 
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
    plt.title('Opencl Transpose Times')
    plt.ylabel('Time')
    plt.xlabel('Iteration')
    plt.savefig('opencl_transpose.png')
    
    
    print "-"*80
    print "cpu"
    print transposed
    print "-"*80
    print "kernel"
    print result
    print "-"*80
    print (transposed==result)
    
    return (transposed==result)
    
#transpose_check() 
    

    
#%%
"""


MATRIX MULTIPLICATION




"""
#%%
#
#
class MatrixMultiply:
    
    def matrix_mul_naive(self, a_cpu):

        print("-"*80)
        print("naive")
        print("-"*80)
        
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # Set up a command queue:
        ctx = cl.Context(devs)
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        a = np.mat(a_cpu).astype(np.float32)

        height = np.array((a_cpu.shape[0])).astype(np.int32)
        width = np.array((a_cpu.shape[1])).astype(np.int32)
        
        c = np.zeros((height, height), np.float32) 
        work_group_size = (np.int32(32), np.int32(32))
        NDRange_size = (np.int32(np.ceil(height / 32.0)*32), np.int32(np.ceil(height / 32.0)*32))

        matrix_multiplication_kernel = """
        __kernel void MatMul(__global float* a_gpu, __global float* output, __global int* input_height, __global int* input_width){
        
            int tx = get_global_id(1); 
            int ty = get_global_id(0);
            int Y = input_height[0];
            int X = input_width[0]; 
            
            
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
        
        prg = cl.Program(ctx, matrix_multiplication_kernel).build()
        func = prg.MatMul
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory

        a_gpu = cl.array.to_device(queue, a)
        c_gpu = cl.array.to_device(queue, c)
        height_gpu = cl.array.to_device(queue, height)
        width_gpu = cl.array.to_device(queue, width)
        
        event = func(queue, NDRange_size, work_group_size, a_gpu.data, c_gpu.data, height_gpu.data, width_gpu.data)
        event.wait()
        result = c_gpu.get() 
        
        kernel_time = (event.profile.end - event.profile.start)/(1e10) 
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time
    
 
    #%%
    
    def matrix_mul_optimized1(self, a_cpu):

        print("-"*80)
        print("opt1")
        print("-"*80)
        
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # Set up a command queue:
        ctx = cl.Context(devs)
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        a = np.mat(a_cpu).astype(np.float32)

        height = np.array((a_cpu.shape[0])).astype(np.int32)
        width = np.array((a_cpu.shape[1])).astype(np.int32)
        
        c = np.zeros((height, height), np.float32) 
        work_group_size = (np.int32(32), np.int32(32))
        NDRange_size = (np.int32(np.ceil(height / 32.0)*32), np.int32(np.ceil(height / 32.0)*32))

        grid_size = np.array(np.int32(np.ceil(width / 32.0)*32)).astype(np.int32)
#        print(grid_size)
        matrix_multiplication_kernel = """
            #define tile_size 32
            
            __kernel void MatMul(__global float* a_gpu, __global float* output, __global int* input_height, __global int* input_width, __global int* grid_size){
                  
            int tx = get_global_id(1); 
            int ty = get_global_id(0);
            const int Y = input_height[0];
            const int X = input_width[0]; 
            const int grid_x = grid_size[0]; 
            
            int worker_y = get_local_id(0);
            int worker_x = get_local_id(1);
   
            __local float A_shared[tile_size][tile_size];  

   
            float summation = 0;   

            for(int i=0; i < grid_x ; i++)
                {
                 if((i * tile_size + worker_x < X) && (ty < Y)) {
                    A_shared[worker_y][worker_x] = a_gpu[ty * X + i * tile_size + worker_x];
                    }
                 else {
                    A_shared[worker_y][worker_x] = 0;
                    }

                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);

                for(int j = 0; j < tile_size; j++)
                    {
                    summation += A_shared[worker_y][j] * a_gpu[j + i * tile_size + tx * X]; 
                    }
                    
                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);
                }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);


            if((ty < Y) && (tx < Y))
                {
                output[ty * Y + tx] = summation;
                }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);
            }         
            

        """
        
        prg = cl.Program(ctx, matrix_multiplication_kernel).build()
        func = prg.MatMul
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory

        a_gpu = cl.array.to_device(queue, a)
        c_gpu = cl.array.to_device(queue, c)
        height_gpu = cl.array.to_device(queue, height)
        width_gpu = cl.array.to_device(queue, width)
        grid_size_gpu = cl.array.to_device(queue, grid_size)
        
        event = func(queue, NDRange_size, work_group_size, a_gpu.data, c_gpu.data, height_gpu.data, width_gpu.data, grid_size_gpu.data)
        event.wait()
        result = c_gpu.get() 
        
        kernel_time = (event.profile.end - event.profile.start)/(1e10) 
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time
        

#%%
#
    def matrix_mul_optimized2(self, a_cpu):

        print("-"*80)
        print("opt2")
        print("-"*80)
        
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # Set up a command queue:
        ctx = cl.Context(devs)
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        a = np.mat(a_cpu).astype(np.float32)

        height = np.array((a_cpu.shape[0])).astype(np.int32)
        width = np.array((a_cpu.shape[1])).astype(np.int32)
        
        c = np.zeros((height, height), np.float32) 
        work_group_size = (np.int32(32), np.int32(32))
        NDRange_size = (np.int32(np.ceil(height / 32.0)*32), np.int32(np.ceil(height / 32.0)*32))

        grid_size = np.array(np.int32(np.ceil(width / 32.0)*32)).astype(np.int32)
#        print(grid_size)
        matrix_multiplication_kernel = """
            #define tile_size 32
            
            __kernel void MatMul(__global float* a_gpu, __global float* output, 
                                 __global int* input_height, __global int* input_width, __global int* grid_size){
                  
            int tx = get_global_id(1); 
            int ty = get_global_id(0);
            const int Y = input_height[0];
            const int X = input_width[0]; 
            
            int worker_y = get_local_id(0);
            int worker_x = get_local_id(1);
            const int grid_x = grid_size[0]; 
   
            __local float A_shared[tile_size][tile_size];  
            __local float B_shared[tile_size][tile_size]; 

   
            float summation = 0;   

            for(int i=0; i < grid_x ; i++)
                {
                 if((i * tile_size + worker_x < X) && (ty < Y)) {
                    A_shared[worker_y][worker_x] = a_gpu[ty * X + i * tile_size + worker_x];
                    }
                 else {
                    A_shared[worker_y][worker_x] = 0;
                    }
                 
                 if((i * tile_size + worker_y < X) && (tx < Y)){
                    B_shared[worker_y][worker_x] = a_gpu[tx * X + i * tile_size + worker_y];
                    }
                 else
                    {
                    B_shared[worker_y][worker_x] = 0;
                    }

                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);

                for(int j = 0; j < tile_size; j++)
                    {
                    summation += A_shared[worker_y][j] * a_gpu[j + i * tile_size + tx * X]; 
                    }
                    
                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);
                }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);


            if((ty < Y) && (tx < Y))
                {
                output[ty * Y + tx] = summation;
                }
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);
            }         
            

        """
        
        prg = cl.Program(ctx, matrix_multiplication_kernel).build()
        func = prg.MatMul
        
        start2 = time.time()
        # transfer host (CPU) memory to device (GPU) memory

        a_gpu = cl.array.to_device(queue, a)
        c_gpu = cl.array.to_device(queue, c)
        height_gpu = cl.array.to_device(queue, height)
        width_gpu = cl.array.to_device(queue, width)
        grid_size_gpu = cl.array.to_device(queue, grid_size)
        
        event = func(queue, NDRange_size, work_group_size, a_gpu.data, c_gpu.data, height_gpu.data, width_gpu.data, grid_size_gpu.data)
        event.wait()
        result = c_gpu.get() 
        
        kernel_time = (event.profile.end - event.profile.start)/(1e10) 
        total_time = time.time() - start2

        """change to just result at some time""" 
        return result, kernel_time, total_time
#
#
#        
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
    
    for i in range(1, 100): 
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
    plt.title('OpenCL Mat Mul Times')
    plt.ylabel('Time')
    plt.xlabel('Iteration')
    plt.savefig('OpenCl_matmul.png')
    
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

