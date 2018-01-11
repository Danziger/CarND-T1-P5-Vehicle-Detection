import cv2
import numpy as np


def features_binned_color(img, size=(32, 32)):
    """
    Computes and returns binned color features.
    
    """

    return cv2.resize(img, size).ravel()


def features_histogram_color(img, bins=32, range=(0, 256)):
    """
    Computes and returns color histogram features.
    
    """
    
    channels = img.shape[-1]
    histograms = []
    
    # Calculate the histograms per channel:    
    for channel in channels:
        histograms.append(np.histogram(img[:, :, channel], bins, range))
    
    # Calculate bins centers:
    edges = histograms[0][1]
    centers = (edges[1:] + edges[0:len(edges) - 1]) / 2
    
    # Concatenate the histograms into a single feature vector:
    features = np.concatenate(histograms)
    
    return features, histograms, centers


def features_hog(img, orients=8, ppc=8, cpb=2, tsqrt=True, norm="L2-Hys", visualise=False, fv=True):
    """
    Returns HOG features and visualization.
    
    """
       
    return hog(
        img,
        orientations=orients,
        pixels_per_cell=(ppc, ppc), 
        cells_per_block=(cpb, cpb),
        transform_sqrt=tsqrt,
        visualise=True,
        feature_vector=fv,
        block_norm=norm
    )

    # return features, hog_image


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        
        # getattr(cv2, "COLOR_RGB2HSV")
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)
        
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
        
    # Return list of feature vectors
    return features

# Lesson 22

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features
