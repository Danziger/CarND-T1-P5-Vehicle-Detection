import cv2
import numpy as np


def draw_boxes(img, boxes, color=(0, 0, 255), alpha=0.5, thick=6):
    """
    Returns a copy of the provided image with boxes (rectangles)
    drawn on it using the specified color and thickness.
    
    """

    overlay = np.copy(img)

    for box in boxes:
        cv2.rectangle(overlay, box[0], box[1], color, thick)
    
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def heatmap_boxes(boxes, height, width):
    """
    Returns a hetmap of size (height, width) based on boxes.
    """

    heatmap = np.zeros((height, width))
    
    for box in boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def heatmap_threshold(heatmap, threshold=1):
    """
    Returns a heatmap with values below threshold filtered out.
    
    """

    heatmap[heatmap <= threshold] = 0
    
    return heatmap