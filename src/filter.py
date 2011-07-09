#!/opt/python2.7/bin/python

import sys
import Image
import ImageChops
import cProfile
import itertools
from abc import ABCMeta, abstractmethod

#import CUDAHandler

# --------------------------------------- #

class Filter:
    """ Clase padre de filtros """
    __metaclass__ = ABCMeta

    # Constantes para los valores minimos y maximos de los pixeles
    MAX_PIXEL_VALUE = 255
    MIN_PIXEL_VALUE = 0
    
    # CUDAHandler
    cuda = None

    # Constantes para indicar con que dispositivo se debe procesar el filtro
    CPU = 0
    CUDA = 1
    OPENCL = 2

    # Diccionario para mapear los modos de procesamiento (CPU, GPU, ...) con
    # los respectivos metodos abstractos que definiran posteriormente los filtros
    processing_method = {} 
        
    # Atributos
    images = []
    post_img = None
    
    def __init__(self, *images):
        for im in images:
            self.images.append(im)
     #   self.cuda = CUDAHandler()

        # El diccionario de metodos se inicializa en el constructor ya que
        # daba problemas crearlo directamente en los atributos
        self.processing_method = { 
            Filter.CPU    : self._processCPU,
            Filter.CUDA   : self._processCUDA,
            Filter.OPENCL : self._processCPU
        }

    # TODO Esquema de colores como parametro
    def newPostImg(self, mode, size):
        self.post_img = Image.new(mode, size)

    def fetchResult(self):
        return self.post_img

    @abstractmethod
    def _processCPU(self):
        pass

    @abstractmethod
    def _processCUDA(self):
        pass

    def Apply(self, mode):
        # Se instancia una nueva imagen postprocesada, que sera la que contenga
        # el resultado de haber aplicado el filtro a la/s imagen/es original/es
        self.newPostImg(self.images[0].mode, (self.images[0].size[0], self.images[0].size[1]))
        # Ahora se llama al metodo de procesado elegido
        self.processing_method[mode]()

# --------------------------------------- #

class ThresholdFilter(Filter):

    # Valor discriminante (menores que este -> MIN_PIXEL_VALUE mayores que este -> MAX_PIXEL_VALUE)
    level = 127

    def __init__(self, *images, **kwargs):
        self.images = []
        super(ThresholdFilter, self).__init__(*images)
        try:
            self.level = kwargs['level']
        except KeyError:
            pass
        
    def _processCPU(self):
        grayscaled = self.images[0].convert("L")
        self.post_img = grayscaled.point(lambda x: Filter.MAX_PIXEL_VALUE if x >= self.level else Filter.MIN_PIXEL_VALUE)

    def _processCUDA(self):
        pass

# --------------------------------------- #

class ErosionFilter(Filter):
    
    # Mascara de aplicacion del filtro; en el constructor de
    # la clase se explica con que se rellena esta lista
    mask = []

    _kernel = """
    #define   ROWS_BLOCKDIM_X 32
    #define   ROWS_BLOCKDIM_Y 8
    #define   ROWS_HALO_STEPS 1

    __global__ void erosion_kernel(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
    ){
        __shared__ float s_Data[ROWS_BLOCKDIM_Y][(2 * ROWS_HALO_STEPS) + ROWS_BLOCKDIM_X];

        //Offset to the left halo edge
        const int baseX = (blockIdx.x * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
        const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

        d_Src += baseY * pitch + baseX;
        d_Dst += baseY * pitch + baseX;

        //Load main data
        #pragma unroll
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
            s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

        //Load left halo
        #pragma unroll
        for(int i = 0; i < ROWS_HALO_STEPS; i++)
            s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

        //Load right halo
        #pragma unroll
        for(int i = 0; i < ROWS_HALO_STEPS; i++)
            s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

        //Compute and store results
        __syncthreads();

        #pragma unroll
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS; i++){
            float result = 0;

            #pragma unroll
            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                result = (s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] > 0) ? 255 : 0; 

            d_Dst[i * ROWS_BLOCKDIM_X] = result;
        }
    }
    """

    def __init__(self, *images):
        self.images = []
        super(ErosionFilter, self).__init__(*images)

        # La mascara contendra tuplas con las sumas que deben efectuarse a la 
        # posicion (x, y) del pixel actual que estamos tratando: por ejemplo, 
        # sumar (1, 0) significa (x + 1, y + 0), osea el pixel situado justo 
        # debajo, y por contra sumar (0, -1) implica (x + 0, y - 1), es decir 
        # el pixel situado justo a la izquierda del actual
        self.mask = [(i, j) for i, j in itertools.permutations([-1, 0, 1], 2) if abs(i) != abs(j)]
        
    def _processCPU(self):
        surrounding_pixels = []
        for row in xrange(self.images[0].size[0]):
            for col in xrange(self.images[0].size[1]):
                try:
                    surrounding_pixels = [self.images[0].getpixel((row + i, col + j)) for i, j in self.mask]
                except IndexError:
                    # Si se produjo alguna excepcion de indexado (debido a que estamos situados en algun borde
                    # de la imagen), construimos la lista de pixels adyacentes replicando el pixel actual
                    # TODO Intentar reconocer que pixel es el que tiro la excepcion, y replicar solo ese
                    surrounding_pixels = list(itertools.repeat(self.images[0].getpixel((row, col)), len(self.mask)))
                finally:
                    # La funcion implicita "all()" devuelve True si _todos_ los elementos de un objeto iterable
                    # son asimismo True (o distintos de 0 si los elementos son numericos, como es el caso)
                    if all(surrounding_pixels):
                        self.post_img.putpixel((row, col), Filter.MAX_PIXEL_VALUE)
                    else:
                        self.post_img.putpixel((row, col), Filter.MIN_PIXEL_VALUE)

    def _processCUDA(self):
        cuda.copyToGPU(self.images[0])
        cuda.setKernel(self._kernel)
        # TODO Dividir los grids atendiendo al ancho y alto de la imagen
        # cuda.Launch(...)
        cuda.getFromGPU()

# --------------------------------------- #

class DifferenceFilter(Filter):
    
    def __init__(self, *images):
        self.images = []
        super(DifferenceFilter, self).__init__(*images)

    def _processCPU(self):
        self.post_img = ImageChops.difference(self.images[0], self.images[1])

    def _processCUDA(self):
        pass

# --------------------------------------- #

if __name__ == '__main__':
    im1 = Image.open(sys.argv[1])
    im2 = Image.open(sys.argv[2])
    print sys.argv[1], ": ", im1.format, im1.size, im1.mode, '\n'

    # Diferencia
    diferencia = DifferenceFilter(im1, im2)
    diferencia.Apply(Filter.CPU)
    tmp = diferencia.fetchResult()

    # Threshold
    threshold = ThresholdFilter(tmp, level=30)
    threshold.Apply(Filter.CPU)
    tmp2 = threshold.fetchResult()

    # Erosion
    erosion = ErosionFilter(tmp2)
    erosion.Apply(Filter.CPU)
    post = erosion.fetchResult()

    post.save("post.png", "PNG")
