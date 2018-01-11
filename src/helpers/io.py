import cv2

import matplotlib.image as mpimg


import src.helpers.constants as C


# TODO: Resize option?


def load_images_rgb(files):    
    return [mpimg.imread(file) for file in files]  


def load_images_as(files, color_space="GRAY"): # HSV, LUV, HLS, YUV, YCR_CB (YCrCb), GRAY
    if color_space not in C.COLOR_SPACES:
        raise ValueError("Invalid color space {}.".format(color_space))
        
    # TODO: Precompute this in a constant?
    CV2_CONVERSION_KEY = getattr(cv2, "COLOR_RGB2" + color_space)
        
    return [cv2.cvtColor(mpimg.imread(file), CV2_CONVERSION_KEY) for file in files]


def load_images_all(files):
    CV2_CONVERSION_KEYS = []
    
    images_per_color_space = [[]]
    color_space_to_index = {}
    index = 1
    
    for COLOR_SPACE in C.COLOR_SPACES:
        CV2_CONVERSION_KEY = getattr(cv2, "COLOR_RGB2" + COLOR_SPACE)
        CV2_CONVERSION_KEYS.append(CV2_CONVERSION_KEY)
        
        images_per_color_space.append([])
        color_space_to_index[CV2_CONVERSION_KEY] = index
        index += 1    
        
    for file in files:
        img_rgb = mpimg.imread(file)
        
        images_per_color_space[0].append(img_rgb)
        
        for CV2_CONVERSION_KEY in CV2_CONVERSION_KEYS:
            images_per_color_space[color_space_to_index[CV2_CONVERSION_KEY]].append(cv2.cvtColor(img_rgb, CV2_CONVERSION_KEY))
            
    return images_per_color_space  # RGB, HSV, LUV, HLS, YUV, YCR_CB (YCrCb), GRAY
