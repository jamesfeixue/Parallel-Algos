import time
import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

np.set_printoptions(threshold='nan')


class MergeSort:

    tile_x = np.int(1024)  # tile size x
    tile_y = np.int(1)  # when change this, remember to change it in kernel!        tile size y

    """

    -------------------------------Serial-------------------------------

    """

    def merge_sort_serial(self, a_cpu):

        # a_cpu: an array generated in cpu.
        # return: the sorted array of a_cpu.
        a_length = len(a_cpu)

        #Base case
        if a_length <= 1:
            return a_cpu

        #Recursive Case
        a_mid = int(a_length/2)
        left = np.array(a_cpu[ 0 : a_mid])
        right = np.array(a_cpu[ a_mid : a_length])

        #Recursively Sort
        merge_sort = MergeSort()
        left = merge_sort.merge_sort_serial(left)
        right = merge_sort.merge_sort_serial(right)

        return merge_sort.merge_serial(left, right)

    def merge_serial(self, left_cpu, right_cpu):
        #initialize
        result = []

        # while not empty
        while (len(left_cpu)>0 and len(right_cpu)>0):
            left_first = left_cpu[0]
            right_first = right_cpu[0]
            # print(left_first, type(left_first), right_first, type(right_first))
            if (left_first <= right_first):
                result.append(left_cpu[0])
                left_cpu = np.array(left_cpu[1:len(left_cpu)])
            else:
                result.append(right_cpu[0])
                right_cpu = np.array(right_cpu[1:len(right_cpu)])

        # consume other when one is empty
        if len(left_cpu) == 0:
            result = np.concatenate((result , right_cpu))
        elif len(right_cpu) == 0:
            result = np.concatenate((result, left_cpu))
        else:
            print("length error")

        return result

#%%


    merge_sort_naive_kernel_code = """
    #include <stdio.h>
    #include <math.h>
    __global__ void Merge_sort_naive(float *a, float *a_temp, float *c, const int a_length){
        //-----assuming blocksize is 32-----
        //-----test on array size over 32-----

        //-----initialize-----
        int tx = threadIdx.x;
        int bx = blockDim.x;
        const int a_len = a_length;
        const int block_size = 32; 

        //-----iterate stride and tile_shift-----
        for (int stride =1; stride<a_len; stride*=2){
            int shift_count = (a_len-1)/(block_size*stride*2)+1;
            
            for (int tile_shift= 0; tile_shift < shift_count; tile_shift++){

                int beginning = tx * stride *2 + tile_shift *  stride * 2 * block_size;
                int middle = beginning + stride;
                int end = middle + stride;

                if (beginning>= a_len) continue; 

                //alter middle and end if necessary
                if (end>a_len){
                    end = a_len;
                }

                if (middle > a_len){
                    middle = a_len;
                }

                int temp_distance_1 = middle - beginning;
                int temp_distance_2 = end - middle;

                //merge
                int m = 0;
                int n = 0;

                while (m<temp_distance_1 && n<temp_distance_2){
                    if (a[beginning+m] < a[middle+n]){
                        a_temp[beginning+m+n]=a[beginning+m];
                        m++;
                    }
                    else if (a[beginning+m] >= a[middle+n]){
                        a_temp[beginning+m + n]=a[middle+n];
                        n++;
                    }
                }

                //put in the rest of arr2
                if (n<temp_distance_2){
                    while (n<temp_distance_2){
                        a_temp[beginning+m+n] = a[middle+n];
                        n++;
                    }
                }

                if (m<temp_distance_1){
                    while (m<temp_distance_1){
                        a_temp[beginning+m+n] = a[beginning+m];
                        m++;
                    }
                }
                    
                __syncthreads(); 
                
                for (int j=0; j<end-beginning; j++){
                    a[beginning+j] = a_temp[beginning+j];
                    a_temp[beginning+j] = 0; //set to zero to clean
                }
                
                __syncthreads(); 
                
                //float print_min = a[beginning]; 
                //float print_max = a[end-1]; 
                //printf("%d, %d || %d, %d, %d : %f, %f | \\n", tile_shift, tx, beginning, middle, end, print_min, print_max); 
            }
            
        }

        for (int k=0; k<a_len; k++){
            c[k] = a[k];
        }

    }
    """




#%%



    merge_sort_optimized1_kernel_code = """
    #include <stdio.h>
    __global__ void Merge_sort_optimized1(float *a, float *c, const int a_length)
    {
        //initialize
        int tx = threadIdx.x;
        const int a_len = a_length;
        //printf("%d", tx);

        //-----load a array into shared memory-----
        __shared__ float a_shared[1024];

        if (tx<a_length){
            a_shared[tx] = a[tx];
        }
        __syncthreads();

        //-----sort-----
        //-----set stride-----
        for (int stride = 1; stride < a_len; stride *= 2){
            int beginning = tx * stride *2;
            int middle = beginning + stride;
            int end = middle + stride;

            if(beginning>=a_len) continue;

            //-----watch for edge cases of beginning, middle, or end larger than a_length-----

            //alter middle and end if necessary
            if (end>a_len){
                end = a_len;
            }

            if (middle>a_len){
                middle = a_len;
            }

            int temp_distance_1 = middle-beginning;
            int temp_distance_2 = end - middle;

            //merge
            int m = 0;
            int n = 0;
            float a_temp[1024];

            while (m<temp_distance_1 && n<temp_distance_2){
                if (a_shared[beginning+m] < a_shared[middle+n]){
                    a_temp[beginning + m +n]=a_shared[beginning+m];
                    m++;
                }
                else if (a_shared[beginning+m] >= a_shared[middle+n]){
                    a_temp[beginning + m + n]=a_shared[middle+n];
                    n++;
                }
            }

            //put in the rest of arr2
            if (n<temp_distance_2){
                while (n<temp_distance_2){
                    a_temp[beginning+m+n] = a_shared[middle+n];
                    n++;
                }
            }

            if (m<temp_distance_1){
                while (m<temp_distance_1){
                    a_temp[beginning+m+n] = a_shared[beginning+m];
                    m++;
                }
            }

            //put temp into shared
            for (int j=beginning; j<end; j++){
                a_shared[j] = a_temp[j];
            }

            __syncthreads();

        }

        for (int k=0; k<a_len; k++){
            c[k] = a_shared[k];
        }
    }
    """

#%%
    prg_merge_sort_naive = SourceModule(merge_sort_naive_kernel_code)
    prg_merge_sort_optimized1 = SourceModule(merge_sort_optimized1_kernel_code)

    def __init__(self):
        self.a_gpu = None

    def prepare_data(self, a_cpu):
        if self.a_gpu is None:
            # self.a_gpu = cl.array.to_device(MatrixMultiply.queue, a_cpu)  # send it to device
            self.a_gpu = gpuarray.to_gpu(a_cpu)

#%%

    """

    -------------------------------Naive-------------------------------

    """

    def merge_sort_naive(self, a_cpu):


        self.prepare_data(a_cpu)
        b_cpu = np.empty_like(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        a_length = np.uint32(len(a_cpu))
        if a_length == 1:
            return a_cpu, 0

        c_gpu = gpuarray.empty((a_length), a_cpu.dtype)
        evt = MergeSort.prg_merge_sort_naive.get_function('Merge_sort_naive')

        start = cuda.Event()
        end = cuda.Event()

        start.record()
        evt(self.a_gpu, b_gpu, c_gpu, a_length, block = (32, 1, 1))
        end.record()
        end.synchronize()

        time_naive = start.time_till(end)*1e-3
        c_naive = c_gpu.get()
        return c_naive, time_naive

#%%
    """

    -------------------------------Optimized 1-------------------------------

    """

    def merge_sort_optimized1(self, a_cpu):
        self.prepare_data(a_cpu)
        
        tile_x = np.int(32)
        tile_y = np.int(1)

        a_length = np.uint32(len(a_cpu))
        if a_length == 1:
            return a_cpu, 0

        c_gpu = gpuarray.empty((a_length), a_cpu.dtype) #TODO verify change
        evt = MergeSort.prg_merge_sort_optimized1.get_function('Merge_sort_optimized1')

        start = cuda.Event()
        end = cuda.Event()

        start.record()
        evt(self.a_gpu, c_gpu, a_length, block = (1024, 1, 1))
        end.record()
        end.synchronize()

        time_optimized1 = start.time_till(end)*1e-3
        c_optimized1 = c_gpu.get()
        return c_optimized1, time_optimized1

#%%
    """
    -------------------------------Test and Plot-------------------------------
    """
if __name__ == '__main__':

    times_naive = []
    times_optimized1 = []
    
    times_naive_whole = []
    times_optimized1_whole = []
    times_serial = []
    
    

    """TODO MAKE ARRAY NOT MATRIX"""
    for i in range(1,1024, 20):
        print("-"*10 + "iteration " +str(i)+ " "+ "-"*10)
        X = np.int(i)

        a_cpu = np.random.randint(1, 10, size=(X)).astype(np.float32)

        mergesort = MergeSort()
        
        start_naive1 = time.time() 
        c_naive, time_naive = mergesort.merge_sort_naive(a_cpu)
        time_naive_temp = time.time() - start_naive1
        times_naive_whole.append(time_naive_temp)
        times_naive.append(time_naive)
        
        start_opt1 = time.time() 
        c_optimized1, time_optimized1 = mergesort.merge_sort_optimized1(a_cpu)
        time_opt1_temp = time.time() - start_opt1 
        times_optimized1_whole.append(time_opt1_temp)
        times_optimized1.append(time_optimized1)

        c_built_in = np.sort(a_cpu)
        
        start_serial = time.time() 
        c_serial = mergesort.merge_sort_serial(a_cpu)
        time_serial_temp = time.time() - start_serial 
        times_serial.append(time_serial_temp) 


        """check results"""
        
#        print("original", a_cpu)
#        print("serial", c_serial)
        
        print("is naive correct: ", np.allclose(c_built_in, c_naive))
#        print("result", (c_naive))
#        print("difference", c_built_in - c_naive)
        # print("built in", (c_built_in))
        print("is optimized1 correct:", np.allclose(c_built_in, c_optimized1))

        # print("result:", c_optimized1)
        # print("true:", c_built_in)

        print("is serial correct:", np.allclose(c_built_in, c_serial))

        """print times"""
#        print("time_naive:", time_naive)
        # print("time_optimized1:", time_optimized1)
        # print("time_serial", time_serial)

    """Draw Results"""
    plt.subplot(2, 1, 1)
    plt.title('merge sort performance - kernel only')
    sizes = np.array(range(1, len(times_naive) + 1))
    plt.plot(sizes, times_serial, 'g--', label='serial')
    plt.plot(sizes, times_naive, 'r--', label='Naive')
    plt.plot(sizes, times_optimized1, 'b--', label='Optimized1')
    
    plt.xlabel('size')
    plt.ylabel('run time')
    plt.legend(loc='upper left')
    
    plt.subplot(2, 1, 2)
    plt.title('merge sort performance -  whole')
    sizes = np.array(range(1, len(times_naive) + 1))
    plt.plot(sizes, times_serial, 'g--', label='serial')
    plt.plot(sizes, times_naive_whole, 'r--', label='Naive')
    plt.plot(sizes, times_optimized1_whole, 'b--', label='Optimized1')
    
    plt.xlabel('size')
    plt.ylabel('run times')
    plt.legend(loc='upper left')
    
    plt.savefig('mergesort_cuda.png')
