from glob import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import random as rnd

sys.path.append(os.path.join(os.path.dirname(__file__), 'classes/'))
from Transformation import Transformation
from LaneLines import LaneLines
from Lines import Lines

counter = 0
processor = None
line = {'left':Lines(n_iter=7), 'right':Lines(n_iter=7)}
fail = {'left':2, 'right':2}

def pipeline(img) :
          
    global lane_lines, line, fail
    
    final, line, fail = lane_lines.get_image_with_lanes(img, line, fail, 15, True, False )
    return final

def main():
    global lane_lines
    global processor 

    if (processor == None):
        tmp_image = plt.imread("test_images/straight_lines1.jpg", 1)
        processor = Transformation(True, False, tmp_image) 
    lane_lines = LaneLines(processor)
    white_output = 'test_project_video.mp4'
    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    harder_output = 'test_harder_project_video.mp4'
    clip1 = VideoFileClip("harder_challenge_video.mp4")
    harder_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    harder_clip.write_videofile(harder_output, audio=False)

if __name__ == "__main__" :
    
    main()
