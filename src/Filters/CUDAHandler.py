#!/opt/python2.7/bin/python

import re
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class CUDAHandler:

    def __init__(self):
        self.host = []
        self.gpu = []
        self.out = []
        self.kernel = None
        self.func_name = ""

    def _createGPUArray(self, i):
        try:
            self.host.append(np.array(i).astype(np.float32))
            self.gpu.append(cuda.mem_alloc(self.host[-1].nbytes))
        except ValueError:
            print "[CUDAHandler Error] Invalid datatype. All of its items mus be floats or integers."
            return
        except:
            print "[CUDAHandler Error] An error occurred. GPU array could not be create."
            return

    def loadData(self, **data):
            try:
                for i in data["input"]:
                   self._createGPUArray(i)            
                   cuda.memcpy_htod(self.gpu[-1], self.host[-1])
                for i in data["output"]:
                   self.gpu.append(cuda.mem_alloc(np.array(i).astype(np.float32).nbytes))
                   cuda.memcpy_htod(self.gpu[-1], i) 
            except KeyError:
                print "[CUDAHandler Error] I/O mismatch."
                return
            except:
                print "[CUDAHandler Error] Input data could not be transfer to GPU."
                return

    def getFromGPU(self):
        for i in self.gpu[len(host):]:
            cuda.memcpy_dtoh(self.out, i)        
        return self.out

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
