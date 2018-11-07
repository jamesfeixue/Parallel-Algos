#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 23:56:39 2018

@author: mobeiusprime
"""
#%%
#imports 
print("-----importing-----")
import datetime

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
print("-----imported-----")
#%%

class openclModule:
    def __init__(self, idata):
        # idata: an array of lowercase characters.
        # Get platform and device (complete)
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # TODO:
        # Set up a command queue (complete)
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx)
        
        # host variables (incomplete)
        # N = 16 #get rid of N #deprecate
        self.a = idata #a is a bunch of letters
        #self.b = np.random.rand(N).astype(np.float32) #deprecate
        
        # device memory allocation (incomplete)
        self.a_gpu = cl_array.to_device(self.queue, self.a) 
        # self.b_gpu = cl_array.to_device(self.queue, self.b) #deprecate
        self.c_gpu = cl_array.empty(self.queue, self.a.shape, self.a.dtype)
        
        # kernel code (incomplete)
        self.kernel = """
        __kernel void func(__global char* a, __global char* c) {
                unsigned int i = get_global_id(0);
                c[i] = a[i]-32;
                }
        """
        
        
        
    def runAdd_parallel(self):
        # return: an array containing capitalized characters from idata and running time.
        # TODO:
        # function call
#        print("-----initializing parallel-----")
        prg = cl.Program(self.ctx, self.kernel).build()
        
        start = datetime.datetime.now()
        prg.func(self.queue, self.a.shape, None, self.a_gpu.data, self.c_gpu.data)
        running_time = datetime.datetime.now() - start 
#        print("-----computation complete-----")
        # memory copy to host
        c = self.c_gpu.get()
        
        # Return output and measured time
#        print(c, running_time)
        return c, running_time 
        
    def runAdd_serial(self): #nice, works with: letters_list = np.array(list("abcdefghijklmnopqrstuvwxyz"))
        # return: an array containing capitalized characters from idata and running time.
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
#from string import ascii_lowercase
#import datetime
#
#letters_list = np.array(list("abcdefghijklmnopqrstuvwxyz"))
#letters_array = np.array(ascii_lowercase[0:26]) #split needs delimiter

#serial_test = openclModule(letters_list)
#serial_test.runAdd_serial()
#
#first_test = openclModule(letters_list)
#result_test = first_test.runAdd_parallel() 

#%%
#final testing ----------------

py_times = [] 
parallel_times = []
for itr in range(1, 40):
    idata = np.array(list("abcdefghijklmnopqrstuvwxyz"*itr)) #extend the array
    ##############################################################################################
    #   capitalize idata using your serial and parallel functions, record the running time here  #
    ##############################################################################################
    
    opencl_capitalizer = openclModule(idata)
    py_output, py_runtime_1 = opencl_capitalizer.runAdd_serial() 
    parallel_output, parallel_runtime_1 = opencl_capitalizer.runAdd_parallel()
    py_times.append(py_runtime_1)
    parallel_times.append(parallel_runtime_1)
    
    print('py_output=\n', py_output) # py_output is the output of your serial function
    print('parallel_output=\n', parallel_output) # parallel_output is the output of your parallel function
    
    code_equality_check = (py_output==parallel_output)
    print('Code equality:\t', not (False in code_equality_check))
    print('string_len=', len(idata), '\tpy_time: ', py_times[itr-1], '\tparallel_time: ', parallel_times[itr-1]) 
    # py_time is the running time of your serial function, parallel_time is the running time of your parallel function.

print("----- serial times are: ", py_times, "-----")
print("----- parallel times are: ", parallel_times, "-----")
print("----- complete -----")

