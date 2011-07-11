#!/usr/bin/env python

from filter import Filter

class DifferenceFilter(Filter):
    
    def __init__(self, *images):
        self.images = []
        super(DifferenceFilter, self).__init__(*images)

    def _processCPU(self):
        self.post_img = ImageChops.difference(self.images[0], self.images[1])

    def _processCUDA(self):
        pass
