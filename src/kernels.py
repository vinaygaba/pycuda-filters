#from pycuda.compiler import SourceModule

#kernels = SourceModule("""
#    __global__ void double(float *a) {
#        int idx = threadIdx;
#        a[idx] *= 2;
#    }
#""")

#double = kernels.get_fuction("double")

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

multiply_them = mod.get_function("multiply_them")
