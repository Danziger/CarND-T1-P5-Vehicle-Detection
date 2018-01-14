import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DPI = 96

# Grid helper funcions:

def getGridFor(n_items, n_cols = 4, hspace = 0.2, wspace = 0.05):
    return getGrid(math.ceil(n_items / n_cols), n_cols, hspace, wspace)

def getGrid(n_rows, n_cols = 4, hspace = 0.2, wspace = 0.05):    
    plt.figure(random.randrange(0, 1000), figsize=(n_cols * 8, n_rows * 6))

    return gridspec.GridSpec(n_rows, n_cols, hspace, wspace)


def showAll(imgs, n_cols, cmap=None):
    gs = getGridFor(len(imgs), n_cols)

    index = 0
    
    # Assuming all the images have the same number of channels:
    shape = imgs[0].shape
    
    if len(shape) == 3 and shape[-1] == 1:
        cmap = cmap or "gray"
    
    for img in imgs:
        ax = plt.subplot(gs[int(index / n_cols), index % n_cols])
        
        ax.imshow(img, cmap=cmap)
        ax.set_title("Figure " + str(index + 1))
        
        index = index  + 1

        
def showAllGrid(imgs, n_cols = 4, scale = 0.5):
    n_rows = math.ceil(len(imgs) / n_cols)
    
    height, width = scale * np.array(imgs[0].shape[:-1]) / DPI
    
    extra = (n_rows - 1) * height * 0.125
    
    plt.figure(random.randrange(0, 1000), figsize=(n_cols * width, n_rows * height + extra), dpi=DPI)

    gs = gridspec.GridSpec(n_rows, n_cols, 0, 0)
    gs.update(wspace=0, hspace=0.125)
           
    
    for i in range(len(imgs)):
        ax = plt.subplot(gs[i])
        # plt.axis('on')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(imgs[i], cmap="gray" if imgs[i].shape[-1] == 1 else None)
        #ax.set_xlim([0, imgs[0].shape[1]])
        
        
def showAllNew(imgs, n_cols = 4, scale = 0.5):
    # https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    # https://stackoverflow.com/questions/8218608/scipy-savefig-without-frames-axes-only-content/8218887#8218887
    # https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
    
    n_rows = math.ceil(len(imgs) / n_cols)
    
    height, width = scale * np.array(imgs[0].shape[:-1]) / DPI
    
    extra = (n_rows - 1) * height * 0.125
    
    fig, subplots = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(n_cols * width, n_rows * height + extra), dpi=DPI)
    
    fig.subplots_adjust(hspace=0.125, wspace=0)

    for ax, img in zip(np.concatenate(subplots), imgs):
        ax.imshow(img, cmap="gray" if img.shape[-1] == 1 else None)
        # ax.set_xlim([0, imgs[0].shape[1]])
        # ax.xaxis.set_visible(False)
