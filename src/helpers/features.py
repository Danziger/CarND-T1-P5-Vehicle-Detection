import cv2
import numpy as np

from skimage.feature import hog


# BINNED COLOR:

def extract_binned_color(imgs, size=(32, 32)):
    return [features_binned_color(img, size) for img in imgs]


def features_binned_color(img, size=(32, 32)):
    """
    Computes and returns binned color features.
    
    """

    return cv2.resize(img, size).ravel()


# HISTOGRAM COLOR:

def extract_histogram_color(imgs, bins=32, bins_range=(0, 256)):
    return [features_histogram_color(img, bins, bins_range)[0] for img in imgs]


def features_histogram_color(img, bins=32, bins_range=(0, 256)):
    """
    Computes and returns color histogram features.
    
    """
        
    channels = img.shape[-1]
    histograms = []
        
    # TODO: Equalize hist?
    
    # Calculate the histograms per channel:    
    for channel in range(channels):
        histogram, edges = np.histogram(img[:, :, channel], bins, bins_range)
        histograms.append(histogram)
    
    # Calculate bins centers:
    centers = (edges[1:] + edges[0:len(edges) - 1]) / 2
    
    # Concatenate the histograms into a single feature vector:
    features = np.concatenate(histograms)
    
    return features, histograms, centers


# HOG:

def extract_hog(imgs, orients=8, ppc=8, cpb=2, tsqrt=True, norm="L2-Hys"):
    return [features_hog(img, orients, ppc, cpb, tsqrt, norm)[0] for img in imgs]


def features_hog(img, orients=8, ppc=8, cpb=2, tsqrt=True, norm="L2-Hys", visualise=False, fv=True):
    """
    Returns HOG features and visualization.
    
    """
       
    # TODO: Make this faster: https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
        
    channels = img.shape[-1]
    hogs_features = []
    visualizations = []
        
    # Calculate HOG per channel:
        
    if visualise:
        for channel in range(channels):
            hog_features, visualization = hog(
                img[:, :, channel],
                orientations=orients,
                pixels_per_cell=(ppc, ppc), 
                cells_per_block=(cpb, cpb),
                transform_sqrt=tsqrt,
                visualise=True,
                feature_vector=fv,
                block_norm=norm
            )

            hogs_features.append(hog_features)
            visualizations.append(visualization)
    else:
        for channel in range(channels):
            hog_features = hog(
                img[:, :, channel],
                orientations=orients,
                pixels_per_cell=(ppc, ppc), 
                cells_per_block=(cpb, cpb),
                transform_sqrt=tsqrt,
                visualise=False,
                feature_vector=fv,
                block_norm=norm
            )

            hogs_features.append(hog_features)

    # Concatenate each channel's hog features into a single feature vector:
    features = np.ravel(hogs_features)
    
    return features, hogs_features, visualizations


# COMBINING FEATURES:

def combine_features(features_list):
    return np.column_stack(features_list)
