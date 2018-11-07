#!/usr/bin/env python3

"""
.
.
.
Python Code
.
.
.
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:57:31 2018

@author: james.f.xue

pyCUDA only
"""
#%%
#Imports
import numpy as np
import math
from pycuda import driver, compiler, gpuarray, tools
from string import ascii_lowercase
import datetime

#initialize the device
import pycuda.autoinit

#%%
#Create Array
#from string import ascii_lowercase
#import datetime
#
#letters_list = list(ascii_lowercase[0:26])
#letters_array = np.array(ascii_lowercase[0:26]) #split needs delimiter

#%%
class cudaModule:
    def __init__(self, idata):
        # idata: an array of lower characters.
        # TODO:
        # Declare host variables
        self.a = idata
        
        # Device memory allocation
        self.N = len(idata)
        self.num_blocks = self.N/1024
        self.output_gpu = gpuarray.empty(self.N * idata.dtype.itemsize, idata.dtype) #allocate for output
        
        
        # Kernel code
        self.capitalize_kernel = """
        __global__ void Capitalize(char *a, char *c)
        {
        
        // ID Thread ID
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; 
        
        // Capitalize through ascii shift
        c[i] = a[i]-32;
        
        }
        """

    def runAdd_parallel(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # Memory copy to device
        input_gpu = gpuarray.to_gpu(self.a) #allocate for input
#        print("Parallel Input completed")
        
        # Function call and measuring time here
        
#        print("Starting Kernel")
        kernel_code = self.capitalize_kernel #get kernel code from template
        mod = compiler.SourceModule(kernel_code) #compile the kernel code
        capitalize = mod.get_function("Capitalize") # get the kernel function
        
        start = datetime.datetime.now() 
        capitalize(input_gpu, self.output_gpu, block=(1024, 1, 1), grid=(self.num_blocks+1,1,1)) 
        total_time = datetime.datetime.now() - start
        
        # Memory copy to host
        result = self.output_gpu.get() 
        
        # Return output and measured time
#        print(result, total_time)
        return result, total_time
        
    def runAdd_serial(self): #so this works
        output_list = []
#        print("-----self.a: ", self.a, " -----")
        letters = self.a 
        
        start = datetime.datetime.now()
        for i in letters: 
            ascii_code = ord(i) #ord take a str of length 1 only
            cap_ascii_code = ascii_code - 32 #shift to capitalize
            cap = chr(cap_ascii_code) #chr transforms the code to a str
            output_list.append(cap)
        running_time = datetime.datetime.now()-start
        
#        print(output_list, running_time)
        return output_list, running_time
        
#%%
#inital testing -------------  
#Create Array

#
#letters_list = np.array(list("abcdefghijklmnopqrstuvwxyz"))
#letters_array = np.array(ascii_lowercase[0:26]) #split needs delimiter
#
#serial_test = cudaModule(letters_list)
#serial_test.runAdd_serial()
#
#first_test = cudaModule(letters_list)
#result_test = first_test.runAdd_parallel() 

#%%

py_times = [] 
parallel_times = []
for itr in range(1, 40):
    idata = np.array(list("abcdefghijklmnopqrstuvwxyz"*itr)) #extend the array
    ##############################################################################################
    #   capitalize idata using your serial and parallel functions, record the running time here  #
    ##############################################################################################
    
    cuda_capitalizer = cudaModule(idata)
    py_output, py_runtime_1 = cuda_capitalizer.runAdd_serial() 
    parallel_output, parallel_runtime_1 = cuda_capitalizer.runAdd_parallel()
    py_times.append(py_runtime_1)
    parallel_times.append(parallel_runtime_1)
    
    print('py_output=\n', py_output, len(py_output)) # py_output is the output of your serial function
    print('parallel_output=\n', parallel_output, len(parallel_output)) # parallel_output is the output of your parallel function
    code_equality_check = (py_output==parallel_output)
    print('Code equality:\t', not (False in code_equality_check))
    print('string_len=', len(idata), '\tpy_time: ', py_times[itr-1], '\tparallel_time: ', parallel_times[itr-1]) 
    # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.

print("----- serial times are: ", py_times, "-----")
print("----- parallel times are: ", parallel_times, "-----")
print("----- complete -----")
