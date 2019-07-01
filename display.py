#!/usr/bin/env python3
"""
This module contains functions for displaying final and intermediate results
when computing segmentation and measurements of bicontinuous nanostructured
materials or dealloyed materials.
"""

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.ndimage.morphology as morphology
import skimage.measure as measure
from skimage import img_as_ubyte
from scipy.stats import lognorm, norm
from skimage.color import gray2rgb, rgb2gray

__author__ = "Joshua Stuckner"
__all__ = ['showFull', 'showOtsu', 'showSkel', 'showHist', 'overlayMask']

def showFull(img, title=None, cmap=None, interpolation='none'):
    """
    Displays a full screen figure of the image.

    Parameters
    ----------
    img : ndarray
        Image to display.
    title : str, optional
        Text to be displayed above the image.
    cmap : Colormap, optional
        Colormap that is compatible with matplotlib.pyplot
    interpolation : string, optional
        How display pixels that lie between the image pixels will be handled.
        Acceptable values are ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’,
        ‘spline16’, ‘spline36’, ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’,
        ‘quadric’, ‘catrom’, ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    """

    # Show grayscale if cmap not set and image is not color.
    if cmap is None and img.ndim == 2:
        cmap = plt.cm.gray
        
    plt.imshow(img, cmap = cmap, interpolation=interpolation)
    plt.axis('off')
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except AttributeError: # TkAgg backend
        figManager.window.state('zoomed')  
    if title is None:
        plt.gca().set_position([0, 0, 1, 1])
    else:
        plt.gca().set_position([0, 0, 1, 0.95])
        plt.title(title)
    plt.ion()
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()



def showOtsu(img, local_otsu, global_otsu, radius, threshold_global_otsu):
    """
    Displays an image and the results of segmentation via regular Otsu's
    method and 2D Otsu's Method along with the threshold value at each pixel
    for direct comparison.  Taken from scikit-image.org

    Parameters
    ----------
    img : image array
        Original image before segmentation.
    local_otsu : 2D array
        Array containing threshold value at each pixel determined by 2D Otsu's
        Method.
    global_otsu : 2D binary array
        Segmentation of the original image by Otsu's Method.
    radius : float
       Radius of neighborhood used for 2D Otsu's Method.
    threshold_global_otsu: int
       Global threshold value determined by Otsu's Method.
    """

    fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True,
                           sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax1, ax2, ax3, ax4 = ax.ravel()

    fig.colorbar(ax1.imshow(img, cmap=plt.cm.gray),
                 ax=ax1, orientation='horizontal')
    ax1.set_title('Original')
    ax1.axis('off')

    fig.colorbar(ax2.imshow(local_otsu, cmap=plt.cm.gray),
                 ax=ax2, orientation='horizontal')
    ax2.set_title('Local Otsu (radius=%d)' % radius)
    ax2.axis('off')

    ax3.imshow(img >= local_otsu, cmap=plt.cm.gray)
    ax3.set_title('Original >= Local Otsu' % threshold_global_otsu)
    ax3.axis('off')

    ax4.imshow(global_otsu, cmap=plt.cm.gray)
    ax4.set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
    ax4.axis('off')

    plt.axis('off')
    plt.ion()
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()


    plt.close()


def showSkel(skeleton, mask, dialate=False, title=None, returnSkel=False,
             cmap=None):
    """
    Displays skelatal data on top of an outline of a binary mask. For example,
    displays a medial axis transform over an outline of segmented ligaments.

    Parameters
    ----------
    skeleton : 2D array
        Data to be displayed.
    mask : binary 2D array
        Mask of segmentation data, the outline of which is displayed along with
        the skel data.
    dialate : boolean, optional
        If dialate is true, the skelatal data will be made thicker in the
        display.
    title : str, optional
        Text to be displayed above the image.
    """

    skel = np.copy(skeleton)

    # Fix error from matplotlib update
    if cmap is None:
        try:
            cmap = plt.cm.spectral
        except AttributeError:
            cmap = plt.cm.nipy_spectral


    # Find the outlines of the mask and make an outline mask called outlines.
    contours = measure.find_contours(mask, 0.5)
    outlines = np.zeros((mask.shape), dtype='uint8')
    for n, contour in enumerate(contours):
        for i in range(len(contour)):
            outlines[int(contour[i,0]), int(contour[i,1])] = 255

    # Make the skel data thicker if dialate is true.
    if dialate:
        skel = morphology.grey_dilation(skel, size=(3,3))


    # Scale the skel data to uint8 and add the outline mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skel = skel.astype(np.float32) # convert to float
        skel -= skel.min() # ensure the minimal value is 0.0
        if skel.max() != 0:
            skel /= skel.max() # maximum value in image is now 1.0
    skel = np.uint8(cmap(skel)*255) # apply colormap to skel data.
    for i in range(3):
        skel[:,:,i] += outlines

    if returnSkel:
        return skel
    
    # Display the results.
    plt.imshow(skel, cmap = cmap, interpolation='none')
    plt.axis('off')
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except AttributeError: # TkAgg backend
        figManager.window.state('zoomed')
    if title is None:
        plt.gca().set_position([0, 0, 1, 1])
    else:
        plt.gca().set_position([0, 0, 1, 0.95])
        plt.title(title)
    plt.ion()
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()




def showHist(data, title=None, xlabel=None,
             numBins=None, gauss=False, log=False):
    """
    Displays a histogram of data with no y-axis and the option to fit gaussian
    and lognormal distribution curves to the data.

    Parameters
    ----------
    data : ndarray
        Data or measurements from which to produce a histogram.
    title : str, optional
        Title of histogram which is displayed at the top of the figure.
    xlabel : str, optional
        Title of the x-axis.  Usually what is measured along the x-axis along
        with the units.
    numBins : int, optional
        Number of possible bars in the histogram.  If not given, the function
        will attempt to automatically pick a good value.
    gauss: boolean, optional
        If true, a fitted guassian distribution curve will be plotted on the
        histogram.
    log: boolean, optional
        If true, a fitted lognormal distribution curve will be plotted on the
        histogram.
    """

    # Determine optimum number of bins.
    if numBins is None:
        u = len(np.unique(data))
        numBins = int(2*u**(1/2))
        if numBins < 4:
            numBins = len(np.unique(data))
            if numBins < 1:
                numBins = 1

    # Create the histogram.
    try:
        n, bins, patches = plt.hist(
            data, bins=numBins, density=1, edgecolor='black')
    except:
        n, bins, patches = plt.hist(
            data, bins=numBins, normed=1, edgecolor='black')
    if log:
        try:
            logfit = lognorm.fit(data.flatten(), floc=0)
            pdf_plot = lognorm.pdf(bins, 
                                   logfit[0], loc=logfit[1], scale=logfit[2])
            plt.plot(bins, pdf_plot, 'r--', linewidth=3, label='lognorm')
        except ValueError:
            pass
    if gauss:
        gfit = norm.fit(data.flatten())
        gauss_plot = norm.pdf(bins, gfit[0], gfit[1])
        plt.plot(bins, gauss_plot, 'g--', linewidth=3, label='gaussian')

    # Display the histogram.
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    ax = plt.gca()
    ax.yaxis.set_visible(False)
    if gauss or log:
        plt.legend()
    plt.ion()
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()


  
def overlayMask(image, mask, color='o', return_overlay=False, animate=False,
                title=None):
    '''
    Displays the binary mask over the original image in order to verify results.
    
    Parameters
    ----------
    image : image array
        Image data prior to segmentation.
    mask : binary array
        Binary segmentation of the image data.  Must be the same size as image.
    color : str, optional
        The color of the overlaid mask.
    return_overlay : bool, optional
        If true, the image with the overlaid mask is returned and the overlay
        is not displayed here.
    animate : bool, optional
        If true, an animated figure will be displayed that alternates between
        showing the raw image and the image with the overlay.

    Returns
    -------
    overlay : RGB image array, optional
        Color image with mask overlayyed on original image (only returned
        if 'return_overlay' is True).
    '''

    if title is None:
        title = 'Segmentation mask overlayed on image'

    img = np.copy(image)

    # Convert the image into 3 channels for a colored mask overlay
    overlay = gray2rgb(img)

    # Set color (default to blue if a proper color string is not given).
    r = 0
    g = 0
    b = 255
    if color == 'red' or color == 'r':
        r = 255
        g = 0
        b = 0
    if color == 'green' or color == 'g':
        r = 0
        g = 255
        b = 0
    if color == 'blue' or color == 'b':
        r = 0
        g = 0
        b = 255
    if color == 'white' or color == 'w':
        r = 255
        g = 255
        b = 255
    if color == 'yellow' or color == 'y':
        r = 255
        g = 255
        b = 0
    if color == 'orange' or color == 'o':
        r = 255
        g = 128
        b = 0
        
    # Apply mask.
    if r != 0:
        overlay[mask == 1, 0] = r
    if g != 0:
        overlay[mask == 1, 1] = g
    if b != 0:
        overlay[mask == 1, 2] = b

    # Return or show overlay.
    if return_overlay:
        return overlay
    else:
        if animate:
            fig = plt.figure()
            ims = []
            ims.append([plt.imshow(image, cmap=plt.cm.gray, animated=True)])
            ims.append([plt.imshow(overlay, animated=True)])
            ani = animation.ArtistAnimation(fig, ims, 1000, True, 1000)
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            try:
                figManager.window.showMaximized()
            except AttributeError: # TkAgg backend
                figManager.window.state('zoomed')
            plt.gca().set_position([0, 0, 1, 0.95])
            plt.title(title)
            fig.canvas.set_window_title('Animated Mask Overlay')
            plt.ion()
            plt.show()
            while plt.get_fignums():
                try:
                    plt.pause(0.1)
                except:
                    pass
            plt.ioff()

          
        else:           
            showFull(overlay, title=title,interpolation='nearest')


