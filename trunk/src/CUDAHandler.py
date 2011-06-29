import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class CUDAHandler:

    # Returns a GPU array
    def _createGPUArray(a):
        try:
            data = np.array(a).astype(np.float32)
        except ValueError:
            print "Invalid list.",
            print "All of its items must be float of integer."
            break
        finally:
            data_gpu = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(data_gpu, data)
            return data_gpu

def copyToGPU(a):
try:
   createGPUArray(a)
except:
raise "FATAL ERROR: GPU array couldn't be create
