import cv2
import numpy as np


def slide_window(
    x0=0,
    x1=None,
    y0=0,
    y1=None,
    size=(64, 64),
    overlap=(0.75, 0.75)):
      
    if x1 is None or y1 is None:
        raise ValueError()
        
    # Compute the span of the region to be searched    
    rangex = x1 - x0
    rangey = y1 - y0
        
    # Compute the number of pixels per step in x/y
    stepx = int(size[0] * (1 - overlap[0]))
    stepy = int(size[1] * (1 - overlap[1]))

    # Compute the number of windows in x/y
    bufferx = int(size[0] * overlap[0])
    buffery = int(size[1] * overlap[1])
    
    countx = int((rangex - bufferx) // stepx)
    county = int((rangey - buffery) // stepy)

    windows = []

    for i in range(countx):
        startx = int(x0 + stepx * i)
        endx = int(startx + size[0])
        
        for j in range(county):
            starty = int(y0 + stepy * j)
            endy = int(starty + size[1])
            
            windows.append(((startx, starty), (endx, endy)))

    return windows


def combine_windows(windows_list=[]):
    return np.concatenate(windows_list)
