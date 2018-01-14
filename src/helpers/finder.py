import cv2
import numpy as np

from scipy.ndimage.measurements import label


import src.helpers.features as FT


def find_cars(img, windows, svc):
    """
    Extracts features from an image using hog sub-sampling and makes predictions about each window's features.
    
    """
   
    detections = []
    
    for start, end in windows:
        x0, y0 = start
        x1, y1 = end
            
        # Extract the image patch
        tile = cv2.resize(img[y0:y1, x0:x1], (64, 64))

        # print(tile[0,0], np.amin(tile[0,0]), np.amax(tile[0,0]))
        
        # Get color features
        hog_features = FT.features_hog(tile, 9, 12, 2)[0]
        spatial_features = FT.features_binned_color(tile, size=(8, 8))          
        hist_features = FT.features_histogram_color(tile, bins=32)[0]
        
        # Scale features and make a prediction
        test_features = np.concatenate([spatial_features, hist_features, hog_features]) 
        
        # test_prediction = svc.predict(test_features.reshape(1, -1))
        test_decision = svc.decision_function(test_features.reshape(1, -1))
 
        # test_prediction is actually test_decision > 0

        # print(test_prediction, test_decision)
        
        if test_decision >= 0.6:
            detections.append((start, end))
                  
    return detections


def find_boxes(heatmap):
    """
    Extracts bounding boxes from a heatmap.
    
    """
    
    boxes = []
    labels = label(heatmap)
    
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        boxes.append(( (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)) ))
        
    return boxes