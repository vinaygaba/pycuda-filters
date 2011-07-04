#!/opt/python2.7/bin/python

import re
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class CUDAHandler:

    data_host = None
    data_cuda = None
    kernel = None
    func_name = ""

    def __init__(self):
        pass

    def copyToGPU(self, a):
        try:
            self.data_host = np.array(a).astype(np.float32)
        except ValueError:
            print "[CUDAHandler Error] Invalid datatype: All items must be integer or float"
            raise
        else:
            self.data_cuda = cuda.mem_alloc(self.data_host.nbytes)
            cuda.memcpy_htod(self.data_cuda, self.data_host)

    def getFromGPU(self):
        processed_data = np.zeros_like(self.data_host)
        cuda.memcpy_dtoh(processed_data, self.data_cuda)
        return processed_data 

    def setKernel(self, kernel_str):
        self.kernel = SourceModule(kernel_str)
        regex = re.compile(r"\s*__global__\s+\w+\s+(\w+)")
        match = regex.search(kernel_str)
        try:
            self.func_name = match.groups()[0]
        except AttributeError:
            print "[CUDAHandler Error] Could not retrieve main kernel function name"
            raise

    # TODO Pasar por argumento un **kwargs directamente, serian:
    #      - hilos por bloque
    #      - numero de bloques (bloques por grid)
    #      - shape del bloque (tupla (x, y, z))
    #      - shape del grid (tupla (x, y))
    # Mirar lo de prepared_call() en vez de get_function()
    def Launch(self, threads_per_block, nblocks):
        func = self.kernel.get_function(self.func_name)
