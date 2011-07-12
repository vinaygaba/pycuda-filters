#!/usr/bin/env python

import cv

from videoHandler import VideoHandler
from Filters.filter import Filter
from Filters.erosion import ErosionFilter
from Filters.difference import DifferenceFilter
from Filters.threshold import ThresholdFilter

class MotionDetector:
   
   pre_video = None
   post_video = None

   def __init__(self, video_path):
      self.pre_video = VideoHandler(video_path)
      self.post_video = VideoHandler()

   def LaunchCPU(self):
      

   def LaunchCUDA(self):
      pass

