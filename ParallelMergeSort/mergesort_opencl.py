"""
Created on Tue Nov  6 17:37:10 2018

@author: james.f.xue
"""
#%%
import time
import pyopencl as cl
import pyopencl.array
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


#%%

class MergeSort: 
    
    NAME = 'NVIDIA CUDA'
    platforms = cl.get_platforms()
    devs = None
    for platform in platforms:
        if platform.name == NAME:
            devs = platform.get_devices()
    ctx = cl.Context(devs)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    tile_x = np.int(32)
    tile_y = np.int(1)
    
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
    __kernel void Merge_sort_naive(__global float* a, __global float* a_temp, __global float* c, const unsigned int a_length)
    {
        //-----initialize-----
        int tx = get_local_id(0);
        int bx = get_group_id(0); 
        int col = bx * get_local_size(0) + tx; 
        
        const int a_len = a_length;
        const int block_size = 32; 

        //-----iterate stride and tile_shift-----
        for (int stride =1; stride<a_len; stride*=2){
            int shift_count = (a_len-1)/(block_size*stride*2)+1;
            
            for (int tile_shift= 0; tile_shift < shift_count; tile_shift++){

                int beginning = col * stride *2 + tile_shift *  stride * 2 * block_size;
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
                    
                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);
                
                for (int j=0; j<end-beginning; j++){
                    a[beginning+j] = a_temp[beginning+j];
                    a_temp[beginning+j] = 0; //set to zero to clean
                }
                
                float min_temp = a[beginning]; 
                float max_temp = a[end]; 
                //printf("%d, %d, %d | %f, %f  \\n", beginning, middle, end, min_temp, max_temp); 
                
                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);
                
            }
            
        }

        for (int k=0; k<a_len; k++){
            c[k] = a[k];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);
    
    }

    """
    #%%
    merge_sort_optimized1_kernel_code = """
    __kernel void Merge_sort_optimized1(__global float* a, __global float* c, const unsigned int a_length)
    {
        //initialize
        int tx = get_local_id(0);
        int bx = get_group_id(0); 
        int col = bx * get_local_size(0) + tx; 
        const int a_len = a_length;

        //-----load a array into shared memory-----
        __local float a_shared[1024];

        if (col<a_length){
            a_shared[col] = a[col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_GLOBAL_MEM_FENCE);

        //-----sort-----
        //-----set stride-----
        for (int stride = 1; stride < a_len; stride *= 2){
            int beginning = col * stride *2; //test
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
            
            float min_temp = a_shared[beginning]; 
            float max_temp = a_shared[end]; 
            // printf("%d, %d, %d | %f, %f  \\n", beginning, middle, end, min_temp, max_temp); 
            
            barrier(CLK_LOCAL_MEM_FENCE);
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        for (int k=0; k<a_len; k++){
            c[k] = a_shared[k];
        }
    
    }
    """
    
    #%%
    
    prg_merge_sort_naive = cl.Program(ctx, merge_sort_naive_kernel_code).build()
    prg_merge_sort_optimized1 = cl.Program(ctx, merge_sort_optimized1_kernel_code).build()
    
    #%%
    
    def __init__(self):
        self.a_gpu = None

    def prepare_data(self, a_cpu):
        if self.a_gpu is None:
            self.a_gpu = cl.array.to_device(MergeSort.queue, a_cpu)
            
    #%%
    
    def merge_sort_naive(self, a_cpu): 
        print("-"*80)
        print("Naive")
        
        a_length = len(a_cpu)
        minimum = min(a_length, 32)
        place_holder = a_cpu[0:minimum]
        
        self.prepare_data(a_cpu)
        place_holder_gpu = cl.array.empty(MergeSort.queue, place_holder.shape, a_cpu.dtype)
        c_naive_gpu = cl.array.empty(MergeSort.queue, a_cpu.shape, a_cpu.dtype)
        b_naive_gpu = cl.array.empty(MergeSort.queue, a_cpu.shape, a_cpu.dtype)
        evt = MergeSort.prg_merge_sort_naive.Merge_sort_naive(MergeSort.queue,
                                                              place_holder_gpu.shape, 
                                                              place_holder_gpu.shape,
                                                              self.a_gpu.data, 
                                                              b_naive_gpu.data, 
                                                              c_naive_gpu.data, 
                                                              np.int32(a_length))
        evt.wait()
        time_naive = 1e-10 * (evt.profile.end - evt.profile.start)
        c_naive = c_naive_gpu.get() 
        return c_naive, time_naive
    
    #%%
    
    def merge_sort_optimized1(self, a_cpu): 
        print("-"*80)
        print("Optimized")
        
        """different a_length version"""
#        a_length = np.array((len(a_cpu))).astype(np.int32)
        a_length = len(a_cpu)
        
        self.prepare_data(a_cpu)
        c_optimized_gpu = cl.array.empty(MergeSort.queue, a_cpu.shape, a_cpu.dtype)
        evt = MergeSort.prg_merge_sort_optimized1.Merge_sort_optimized1(MergeSort.queue,
                                                                        c_optimized_gpu.shape, 
                                                                        c_optimized_gpu.shape, 
                                                                        self.a_gpu.data, 
                                                                        c_optimized_gpu.data, 
                                                                        np.int32(a_length))
        evt.wait()
        time_optimized = 1e-10 * (evt.profile.end - evt.profile.start)
        c_optimized = c_optimized_gpu.get() 
        return c_optimized, time_optimized
        
        
    #%%
    
if __name__ == '__main__':
    
    times_naive = []
    times_optimized1 = []
    
    times_naive_whole = []
    times_optimized1_whole = []
    times_serial = []
    
    
    for i in range(1,120000, 1000):
        print("-"*10 + "iteration " +str(i)+ " "+ "-"*10)
        X = np.int(i)

        a_cpu = np.random.randint(1, 10, size=(X)).astype(np.float32)
        mergesort = MergeSort() 
        
        start_naive1 = time.time() 
        c_naive, time_naive = mergesort.merge_sort_naive(a_cpu)
        time_naive_temp = time.time() - start_naive1
        times_naive_whole.append(time_naive_temp)
        times_naive.append(time_naive)
        
#        start_opt1 = time.time() 
#        c_optimized1, time_optimized1 = mergesort.merge_sort_optimized1(a_cpu)
#        time_opt1_temp = time.time() - start_opt1 
#        times_optimized1_whole.append(time_opt1_temp)
#        times_optimized1.append(time_optimized1)

        c_built_in = np.sort(a_cpu)
        
        start_serial = time.time() 
        c_serial = mergesort.merge_sort_serial(a_cpu)
        time_serial_temp = time.time() - start_serial 
        times_serial.append(time_serial_temp) 


        """check results"""
        
#        print(a_cpu)
#        print("difference ", c_built_in-c_optimized1)
#        print(c_optimized1)      
  
        
        """check correctness"""
        print("is naive correct: ", np.allclose(c_built_in, c_naive))
#        print("is optimized1 correct:", np.allclose(c_built_in, c_optimized1))
        print("is serial correct:", np.allclose(c_built_in, c_serial))

    """Draw Results"""
    plt.subplot(2, 1, 1)
    plt.title('merge sort performance - kernel only')
    sizes = np.array(range(1, len(times_optimized1) + 1))
    plt.plot(sizes, times_serial, 'g--', label='serial')
    plt.plot(sizes, times_naive, 'r--', label='Naive')
#    plt.plot(sizes, times_optimized1, 'b--', label='Optimized1')
    
    plt.xlabel('size')
    plt.ylabel('run time')
    plt.legend(loc='upper left')
    
    plt.subplot(2, 1, 2)
    plt.title('merge sort performance -  whole')
    sizes = np.array(range(1, len(times_optimized1_whole) + 1))
    plt.plot(sizes, times_serial, 'g--', label='serial')
    plt.plot(sizes, times_naive_whole, 'r--', label='Naive')
#    plt.plot(sizes, times_optimized1_whole, 'b--', label='Optimized1')
    
    plt.xlabel('size')
    plt.ylabel('run times')
    plt.legend(loc='upper left')
    
    plt.savefig('mergesort_opencl.png')
    
    
    
    
    
    
    