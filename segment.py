#!/usr/bin/env python3
"""
This module contains functions to segment the two phases of a bicontinuous
nanostructured material.  Phases can be solid/pore (nanoporous metals) or
solid/solid (bicontinuous composites).
"""
import warnings

import time
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk, remove_small_objects
from skimage.restoration import denoise_bilateral
from skimage.color import rgb2gray

from aquami import inout
from aquami import display
from aquami import gui

__author__ = "Joshua Stuckner"

def smoothEdges(mask, smooth_radius=1):
    """
    Smoothes the edges of a binary mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    smooth_radius : float64
        The radius of the smoothing operation.  See note below.
        
    Returns
    -------
    smooth_mask : 2D array of bool
        Mask that has been smoothed.

    Notes
    -----
    smooth_radius sets the structure element (selem) for smoothing the edges of
    the masks. If the smooth_rad rounds up the selem is a disk with radius
    rounded up.  If smooth_radius rounds down, selem is a box.

    Radius = 0 - 1.499
    [[1,1,1],
     [1,1,1],
     [1,1,1]]

    Radius = 1.5 - 1.99
    [[0,0,1,0,0],
     [0,1,1,1,0],
     [1,1,1,1,1],   
     [0,1,1,1,0],
     [0,0,1,0,0]]
     
    Radius = 2 - 2.499
    [[1,1,1,1,1],
     [1,1,1,1,1],
     [1,1,1,1,1],   
     [1,1,1,1,1],
     [1,1,1,1,1]]    
    """

    smooth_radius = max(smooth_radius, 1)

    if round(smooth_radius, 0) > int(smooth_radius): # If round up.
        size = int(smooth_radius + 1)
        selem = disk(round(smooth_radius,0))
    else:
        size = 1 + 2*int(smooth_radius)
        selem = np.ones((size,size))

    # Smooth edges.
    # It is necessary to perform this step because skelatonizing algorithms are
    # extremely sensitive to jagged edges and may otherwise give spurious
    # results.
    smooth_mask = ndimage.binary_opening(mask, structure=selem)
    smooth_mask = ndimage.binary_closing(smooth_mask, structure=selem)

    return smooth_mask

def removeSmallObjects(mask, small_objects=0, small_holes=0):
    """
    Removes small objects (white areas of mask) and small holes (black areas).

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    small_objects : int
        Max area of connected white pixels that will be removed.
    small_holes : int
        Max area of connected black pixels that will be removed.        
        
    Returns
    -------
    out_mask : 2D array of bool
        Mask with small holes and objects removed.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_mask = remove_small_objects(mask, small_objects)
        out_mask = ~remove_small_objects(~out_mask, small_holes)

    return out_mask
        
def roughSegment(image):
    """
    Returns binary masks of the segmented bright and dark phases.  Uses Otsu's
    Method and some morphological operations to reduce noise.

    Parameters
    ----------
    image : 2D array of uint8
        Image data to be segmented

    Returns
    -------
    bright : 2D array of bool
        Mask of bright phase.
    dark : 2D array of bool
        Mask of dark phase or pores.
    """

    # Don't operate on the passed image.
    img = np.copy(image)
    
    # Get shape of image and convert to grayscale.
    try:
        rows, cols =  img.shape
    except ValueError: # Convert to grayscale
        img = rgb2gray(img)
        rows, cols =  img.shape

    # Ensure proper datatype.
    img = inout.uint8(img)

    # Set radius of smoothing filter based on the image resolution.
    blur_mult = 1.4 # Increasing this parameter increases the blur radius.
    blur_radius = max(1,(cols*blur_mult/1024))

    # Smooth the image to reduce noise.
    img = ndimage.gaussian_filter(img, blur_radius)

    # Ensure proper datatype after bilateral smoothing.
    img = inout.uint8(img)
    
    # Use Otsu's Method to determine the global threshold value.
    threshold_global_otsu = threshold_otsu(img)
    # Generate masks of both phases based on Otsu's threshold value.
    bright = img >= threshold_global_otsu
    dark = img < threshold_global_otsu

    # Morphological operations to remove noise.
    small = 200 # Area in pixels^2 of objects to remove as noise
    bright = smoothEdges(bright, 1)
    dark = smoothEdges(dark, 1)
    bright = removeSmallObjects(bright, small, small)
    dark = removeSmallObjects(dark, small, small)

    return bright, dark

def segment(image, estSize):
    """
    Returns binary masks of the segmented bright and dark phases.  Uses 
    two-dimentional Otsu's Method and some morphological operations to
    reduce noise.  2D Otsu's Method takes significantly longer to perform than
    the regular method and is more robust against illumination and contrast
    changes throughout the image by setting a seperate threshold value for each
    pixel.

    Parameters
    ----------
    image : 2D array of uint8
        Image data to be segmented
    estSize : float
        Rough estimate of ligament diameter.  Used to set parameters such as
        the blur radius for smoothing.

    Returns
    -------
    bright : 2D array of bool
        Mask of bright phase.
    dark : 2D array of bool
        Mask of dark phase or pores.
    """

    # Don't operate on the passed image.
    img = np.copy(image)

    # Adjustable parameters.
    # These have been highly tested and changing these may throw off results.
    otsuMultiplier = 5 # Otsu neighborhood = otsuMultiplier * estSize
    blur_mult = 1.4 # Increasing this parameter increases the blur radius.
    small_mult = 0.025 # Remove objects with area < small_mult*3.14*estSize^2.
    smooth_div = 36 # Sets the structure element radius for edge smoothing.
    small_blob_mult = 0.25 # Used to remove small blobs after edge smoothing.
    small_hole_mult = 0.05 # Used to remove small holes after edge smoothing.
    

    # Get shape of image and convert to grayscale.
    try:
        rows, cols =  img.shape
    except ValueError: # Convert to grayscale
        img = rgb2gray(img)
        rows, cols =  img.shape

     
    # Smooth the image.
    blur_radius = max(1,int(round(cols*blur_mult/1000,0)))
    img = denoise_bilateral(img,
                            sigma_color=0.05,  #increase this to blur more
                            sigma_spatial=blur_radius,
                            multichannel=False)
 
    # Ensure proper datatype after bilateral smoothing.
    img = inout.uint8(img)

    # Perform Otsu thresholding
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu    

    # Perform 2D Otsu thresholding.
    radius = estSize * otsuMultiplier
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    bright = img >= local_otsu
    dark = img < local_otsu

    
    # Remove tiny objects in both phases.
    small = small_mult * 1/2 * 3.14 * estSize**2
    bright = removeSmallObjects(bright, small, small)
    dark = removeSmallObjects(dark, small, small)

    # Smooth edges.  
    smooth_rad = max(estSize/smooth_div, 1)
    bright = smoothEdges(bright, smooth_rad)
    dark = smoothEdges(dark, smooth_rad)

    # Remove small blobs and really small holes.
    small_blobs = small_blob_mult * 1/2 * 3.14 * estSize**2
    small_holes = small_hole_mult * 1/2 * 3.14 * estSize**2
    bright = removeSmallObjects(bright, small_blobs, small_holes)
    dark = removeSmallObjects(dark, small_blobs, small_holes)
    
    return bright, dark

def manualSegment(image):
    """
    Returns binary masks of the segmented bright and dark phases based on
    manually selected threshold value.

    Parameters
    ----------
    image : 2D array of uint8
        Image data to be segmented

    Returns
    -------
    bright : 2D array of bool
        Mask of bright phase.
    dark : 2D array of bool
        Mask of dark phase or pores.
    """

    # Adjustable parameters.
    # These have been highly tested and changing these may throw off results.
    blur_mult = 1.4 # Increasing this parameter increases the blur radius.
    small_div = 20 # Increasing this reduces hole close size in low res images.

    
    img = inout.uint8(image)

    # Get shape of image and convert to grayscale.
    try:
        rows, cols =  img.shape
    except ValueError: # Convert to grayscale
        img = rgb2gray(img)
        rows, cols =  img.shape

    small = int((min(rows,cols)/small_div)**2)
    if small > 200:
        small = 200

    # Smooth the image to reduce noise.
    blur_radius = max(1,(cols*blur_mult/1024))
    img = ndimage.gaussian_filter(img, blur_radius)

    # Get the default threshold value.
    threshinit = threshold_otsu(img)

    # Launch the GUI for manual threshholding.
    p = gui.manualThreshhold(img, threshinit)
    p.show()
    bright = p.getMask()
    dark = ~bright

    # Morphological operations to remove noise.
    bright = smoothEdges(bright, 1)
    dark = smoothEdges(dark, 1)
    bright = removeSmallObjects(bright, small, small)
    dark = removeSmallObjects(dark, small, small)

    return bright, dark
