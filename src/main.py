#!/usr/bin/env python

import sys
import Image
import ImageChops

from Filters.filter import Filter
from Filters.erosion import ErosionFilter
from Filters.difference import DifferenceFilter
from Filters.threshold import ThresholdFilter

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage:', sys.argv[0], '<image1> <image2'
        sys.exit(1)
    
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

    # TODO Mergeado en una clase aparte
    r, g, b = im2.split()
    tmp = ImageChops.add(r, post)
    merged = Image.merge("RGB", (tmp, g, b))

    merged.save("merged.png", "PNG")
