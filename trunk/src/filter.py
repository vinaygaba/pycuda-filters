#!/opt/python2.7/bin/python

import sys
import Image
import cProfile
from abc import ABCMeta, abstractmethod

import CUDAHandler
from profiling import Profile

# --------------------------------------- #

class Filter:
    """ Clase padre de filtros """
    __metaclass__ = ABCMeta

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
        self.cuda = CUDAHandler()

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

    # Constantes para los valores minimos y maximos de los pixeles
    MAX_PIXEL_VALUE = 255
    MIN_PIXEL_VALUE = 0
    
    # Valor discriminante (menores que este -> 0; mayores que este -> 255)
    threshold = 127

    def __init__(self, *images, **kwargs):
        super(ThresholdFilter, self).__init__(*images)
        try:
            self.threshold = kwargs['threshold']
        except KeyError:
            pass
        
    def _processCPU(self):
        # Modo "1" equivale a threshold con un valor discriminante estandar de 127
        # TODO comprobar si retorna una nueva instancia o si la aplica sobre la misma
        im = self.images[0].convert("1")
        # self.images[0].point(lambda x: MAX_PIXEL_VALUE if x >= self.threshold else MIN_PIXEL_VALUE)

    def _processCUDA(self):
        pass

# --------------------------------------- #

class ErosionFilter(Filter):
    def __init__(self, *images):
        super(ErosionFilter, self).__init__(*images)
        
    def _processCPU(self):
        pass

    def _processCUDA(self):
        pass

# --------------------------------------- #

class DifferenceFilter(Filter):
    
    def __init__(self, *images):
        super(DifferenceFilter, self).__init__(*images)

    def _processCPU(self):
        for x in xrange(self.images[0].size[0]):
            for y in xrange(self.images[0].size[1]):
                # "diff" resultara ser una tupla de 3 elementos (en imagenes RGB) con la 
                # diferencia en valor absoluto por cada canal en ese pixel, comparado con el
                # mismo pixel de la imagen anterior
                diff = tuple([abs(a - b) for a,b in 
                              zip(self.images[0].getpixel((x, y)), self.images[1].getpixel((x, y)))])
                # img.putpixel((x, y), value)
                self.post_img.putpixel((x, y), diff)

    def _processCUDA(self):
        pass

# --------------------------------------- #

im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])
print sys.argv[1], ": ", im1.format, im1.size, im1.mode, '\n'
diferencia = DifferenceFilter(im1, im2)
diferencia.Apply(Filter.CPU)
post = diferencia.fetchResult()
post.save("post.png", "PNG")
