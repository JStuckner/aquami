#!/usr/bin/env python3
"""
This module contains functions to perform measurements on BNMs. Includes
functions for skeletonization, distance transforms, and measuring ligament
length and width.
"""

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.morphology import (skeletonize, remove_small_objects, disk,
                                binary_dilation, binary_opening)
from skimage.measure import label
from scipy.signal import convolve2d
from scipy.stats import lognorm, norm
from scipy.ndimage.morphology import distance_transform_edt
import tkinter as tk

from aquami import display
from aquami import gui
from aquami import inout
from aquami import segment

__author__ = "Joshua Stuckner"

def isConnected(mask):
    """
    Determines whether the masked phase is fully connected (returns True) or
    seperated (returns False)

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.

    Returns
    -------
    bool
        True if mask is mostly connected.  False otherwise.
    """

    # Label each blob.  0 = background, 1 = blobs touching edge.
    lig_labels = label(mask, background=0)

    # Find the area of each connected section.
    areas = np.bincount(lig_labels.flatten()) #count the pixels
    areas[1:][::-1].sort() #sort in descending order (ignoring the background).

    # If connected area touching the edge is larger than the sum of all other
    # areas then this phase is basically fully connected.
    # Use the appropriate skeletonizing function.
    if areas[1] > np.sum(areas[2:]):
        return True
    else:
        return False
    
def skelConnected(mask, diam, removeEdge=False, times=3):
    """
    Returns the skeletal backbone of the ligaments or pores which are one pixel
    thick lines sharing the same connectivity as the passed binary mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase. Must be continuously connected.
    diam : float64
        The estimated or measured average diameter of the ligaments in the mask.
    removeEdge : bool, optional
        If true, the returned skel does not include ligaments that are too close
        to the edge.

    Returns
    -------
    skel : 2D array of bool
        Skeletal backbone of the mask.  One pixel thick lines sharing the same
        connectivity as the mask.
    """

    # Get the shape of the mask.
    rows, cols = mask.shape

    # Find the initial skeletal backbone.
    skel = skeletonize(mask>0)
    
    # Remove  terminal edges.
    
    for i in range(times):
        # Find number of 1st and 2nd neighboring pixels
        neighbors = convolve2d(skel, np.ones((3,3)), mode='same')
        neighbors = neighbors * skel
        neighbors2 = convolve2d(skel, np.ones((5,5)), mode='same')
        neighbors2 = neighbors2 * skel
    
        # Remove nodes and label each ligament section.
        nodes = neighbors > 3
        skel = np.bitwise_xor(skel, nodes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skel = remove_small_objects(skel, 1, connectivity=1)
        labels = label(skel, connectivity=2, background=0)

        # Find terminal sections.
        terminal_labels = labels[neighbors==2] #These are definitly terminal
        # When there is a 1 pixel fork at the end, we need to look at second
        # nearest neighbors as well.
        #neighbors[neighbors2>5] = 10
        #terminal_labels2 = labels[neighbors<5]
        #terminal_labels = np.append(terminal_labels, terminal_labels2)
        terminal_sections = np.unique(terminal_labels)
        just_terminal = np.zeros(skel.shape, dtype=bool)
        for lab in terminal_sections:
            just_terminal[labels==lab] = 1

        # Remove terminal sections. 
        skel = np.bitwise_xor(skel, just_terminal)
        
        # Put the nodes back in.
        skel = binary_dilation(skel, selem=disk(3))    
        skel = skeletonize(skel)

    # Remove sections that touch the edge of image.
    if removeEdge:
        try:
            edge = round(int(diam))
        except ValueError:
            edge = 0
        skel = np.bitwise_xor(skel, nodes)
        labels = label(skel, connectivity=2, background=0)
        elabels = np.copy(labels)
        elabels[edge:rows-edge-1,edge:cols-edge-1] = 0
        edge_labels = np.unique(elabels)
        for lab in edge_labels:
            skel[labels==lab] = 0
            
        # Put the nodes back in.
        skel = skel+nodes    

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skel = remove_small_objects(skel, 8, connectivity=2)

    return skel

def skelBlobs(mask, diam):
    """
    Returns the skeletal backbone of the ligaments or pores which are one pixel
    thick lines sharing the same connectivity as the passed binary mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase. Must be seperated blobs.
    diam : float64
        The estimated or measured average diameter of the ligaments in the mask.

    Returns
    -------
    skel : 2D array of bool
        Skeletal backbone of the mask.  One pixel thick lines sharing the same
        connectivity as the mask.
    """

    # Don't operate on original mask.
    im = np.copy(mask)

    # Smooth edges to reduce spurious results.
    im = binary_opening(im>0, selem=disk(1))

    # Get initial skeletonized image.
    skel = skeletonize(im>0)

    return skel

def skeleton(mask, diam, removeEdge=False, times=3):
    """
    Returns the skeletal backbone of the ligaments or pores which are one pixel
    thick lines sharing the same connectivity as the passed binary mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    diam : float64
        The estimated or measured average diameter of the ligaments in the mask.
    removeEdge : bool, optional
        If true, the returned skel does not include ligaments that are too close
        to the edge.
        
    Returns
    -------
    skel : 2D array of bool
        Skeletal backbone of the mask.  One pixel thick lines sharing the same
        connectivity as the mask.
    """

    if isConnected(mask):
        return skelConnected(mask, diam, removeEdge=removeEdge, times=times)
    else:
        return skelBlobs(mask, diam)

def distanceTransform(mask):
    """
    Returns the euclidean distance each white pixel is to the nearest black
    pixel.  Just calls scipy.ndimage.morphology.distance_transform_edt.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.

    Returns
    -------
    dist : ndarray
        Size in nm of each pixel
    """

    return distance_transform_edt(mask)
    
def manualPixelSize(scale):
    '''
    Returns the size of each pixel in nm.  Must manually select the scale bar
    and input the size.

    Parameters
    ----------
    scale : ndarray
        Image containing the scale bar

    Returns
    -------
    pixelSize : float64
        Size in nm of each pixel
    '''

    plt.imshow(scale, cmap=plt.cm.gray, interpolation=None)
    plt.title("Draw a rectangle the same width as the scale bar and close the"+\
            " figure.")
    figManager = plt.get_current_fig_manager()
    try:
        figManager.window.showMaximized()
    except AttributeError: # TkAgg backend
        figManager.window.state('zoomed')
    a = gui.selectRect()
    plt.axis('off')
    plt.ion()
    plt.show()
    while plt.get_fignums():
        try:
            plt.pause(0.1)
        except:
            pass
    plt.ioff()
    barSize = a.x1-a.x0
    intScaleNum = 0
    root = tk.Tk()
    pixapp = gui.inputScale(root, scale=scale)
    intScaleNum = pixapp.result
    root.destroy()
    try:
        pixelSize = round(intScaleNum/barSize, 2) # More decimals are arbitrary.
    except TypeError:
        pixelSize = 0
    return pixelSize

def estimateDiameter(mask):
    '''
    Returns a rough estimate of the ligament size in order to set parameters
    for more refined thresholding and size calculations.

    Parameters
    ----------
    mask : ndarray
        Binary image file containing thresholded data

    debug : boolean, option
        Set true to display the process

    Returns
    -------
    estSize: float64
        Rough estimate of ligament average radius
    '''
    
    rows, cols = mask.shape
    sizes = np.zeros((2048))
    count = 0 #keeps track of number of pixels

    # Horizontal lines
    for i in range(rows):
        count = 0
        for j in range(cols):
            pixel = mask[i,j]
            if pixel:
                count += 1
            else:
                if count != 0:
                    sizes[count] += 1
                count = 0
            if j == cols - 1 and count != 0:
                sizes[count] += 1

    # Vertical lines
    for j in range(cols):
        count = 0
        for i in range(rows):
            pixel = mask[i,j]
            if pixel:
                count += 1
            else:
                if count != 0:
                    sizes[count] += 1
                count = 0
            if j == cols - 1 and count != 0:
                sizes[count] += 1

    #Calculate average diameter
    sumation = 0
    for i in range(5, len(sizes)):
        sumation += i*sizes[i]
    estSize = sumation / np.sum(sizes)

    return estSize

def diameter(mask, estSize, returnAll=False, showSteps=False, pdf=None):
    """
    Calculates the average ligament size and standard deviation of the passed
    mask.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    estSize : float64
        The estimated diameter.  Used to set some parameters.
    returnAll : bool, optional
        If set to true, a list of all the measurements will be returned.
    showSteps : bool, optional
        If set to true, figures will be displayed that shows the results of
        intermediate steps when calculating the diameter.
    pdf : matplotlib.backends.backend_pdf.PdfPages, optional
        If provided, information will be output to the pdf file.
        
    Returns
    -------
    average : float64
        Average ligament diameter.  (Assumes gaussian distribution.)
    SD : float64
        Standard deviation of ligament diameters.
    allMeasurements: 1D array of float64, optional
        A list of all measured ligament diameters.
    """

    rows, cols = mask.shape
    
    # Adjustable parameters.
    # These have been highly tested and changing these may throw off results.
    try:
        ignore_edge_dist = int(estSize * 3) #Ignores results this close to the edge.
    except ValueError:
        #This only happens if estSize is NaN
        ignore_edge_dist = 70 
    
    # make sure the ignored edge isn't too big
    narrow = min(rows,cols)    
    if ignore_edge_dist > narrow / 3:
        ignore_edge_dist = int(narrow/6)
        
    thresh = estSize / 20 # Diameter's smaller than this are ignored as noise.
    
    # Get the distance transform and the skelatal backbone.
    dist = distanceTransform(mask)
    dist = dist * 2 # Convert radius to diameter.
    skel = skeleton(mask, estSize)
    

    # Need to do more processing on the skel if non connected phase.
    if not isConnected(mask):
        # Remove end-of-line pixels from the skel.
        pixels_to_remove = int(estSize / 2)
        for i in range(pixels_to_remove):
            keep = np.ones(skel.shape)
            terminal_ends = convolve2d(skel, np.ones((3,3)), mode='same')
            keep[terminal_ends < 3] = 0
            skel *= keep > 0

    # Multiply them together to get the diameter measurements along the backbone
    # of each ligament.
    dist *= skel

    if showSteps:
        dialate = False if cols < 1500 else True #Thicker result at high res.
        display.showSkel(dist, mask, dialate=dialate,
                         title="Ligament diameters along the backbone."
                         )
    if pdf is not None:
        dialate = False if cols < 1500 else True #Thicker result at high res.
        inout.pdfSaveSkel(pdf, dist, mask, dialate=dialate,
                         title="Ligament diameters along the backbone."
                         )
    # Remove the edges of the distance map.
    dist = dist[ignore_edge_dist:-ignore_edge_dist,
                ignore_edge_dist:-ignore_edge_dist]

    # Prepare list of all measurements.
    allMeasurements = np.sort(dist[dist>0].flatten())
    
    # Ignore small measurements as noise.
    dist = dist[dist>thresh]

    if showSteps:
        display.showHist(dist, gauss=True, log=True, title='Ligament Diameters',
                         xlabel='Ligament diameter [pixels]')

    if pdf is not None:
        inout.pdfSaveHist(pdf, dist, gauss=True, log=True,
                          title='Ligament Diameters',
                          xlabel='Ligament diameter [pixels]')
        
    average = np.average(dist)
    SD = np.std(dist)
    
    if returnAll:
        return average, SD, allMeasurements
    else:
        return average, SD

def length(mask, estSize, returnAll=False, showSteps=False, pdf=None):
    """
    Calculates the average between node ligament length and standard deviation
    of the passed mask.  Only meaningful on the fully connected phase.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase. 
    estSize : float64
        The estimated diameter.  Used to set some parameters.
    returnAll : bool, optional
        If set to true, a list of all the measurements will be returned.
    showSteps : bool, optional
        If set to true, figures will be displayed that shows the results of
        intermediate steps when calculating the diameter.
    pdf : matplotlib.backends.backend_pdf.PdfPages, optional
        If provided, information will be output to the pdf file.
        
    Returns
    -------
    logavg : float64
        Average ligament length.  (Assumes lognormal distribution.)
    logstd : float64
        Standard deviation of ligament diameters.
    allMeasurements: 1D array of float64, optional
        A list of all measured ligament lengths.
    """

    # Debug mode
    debug = False
    if debug:
        showSteps = True

    # Adjustable parameters.
    # These have been highly tested and changing these may throw off results.
    small_lig = 0.6*estSize # Ligaments smaller than this are within a node.
    filter_out_mult = 0.4 # This * estSize is ignored in final calculation.
    
    # Return 0 if the mask is not fully connected.
    if not isConnected(mask):
        if returnAll:
            return 0,0,0
        else:
            return 0,0
        
    rows, cols = mask.shape
    skel = skeleton(mask, estSize, removeEdge=True, times=3)
    
    if debug:
        display.showSkel(skel, mask)

    # get ligament and node labels
    nodes = np.copy(skel)
    ligaments = np.copy(skel)
    neighbors = convolve2d(skel, np.ones((3,3)), mode='same')
    neighbors = neighbors * skel
    nodes[neighbors < 4] = 0
    ligaments[neighbors > 3] = 0
    ligaments = label(ligaments, background=0)
    nodes = binary_dilation(nodes, selem=disk(3))
    nodes = label(nodes, background=0)    
    
    # get a list of ligaments connected to each node
    node_to_lig = []  # node_to_lig[n] is an array of each ligament label that is connected to node n
    unodes = np.unique(nodes)
    for n in unodes:
        node_to_lig.append(np.unique(ligaments[nodes==n]))

    # get a list of nodes connected to each ligament
    lig_to_node = []  # lig_to_node[l] is an array of each node label that is connected to ligament l
    uligs = np.unique(ligaments)
    for l in uligs:
        lig_to_node.append(np.unique(nodes[ligaments==l]))
        
    # Get the length of each ligament between nodes.
    lengths = np.bincount(ligaments.flatten())

    # Add ligaments that are within a single node to connected ligaments.
    small = int(round(small_lig))
    too_small = []
    for l in uligs:
        if lengths[l] <= small:
            too_small.append(l) #keep track of ligaments that are too small
            for connected in lig_to_node[l]:
                if connected > 0 and connected != l:
                    # add half the small ligament to the connected ligaments.
                    lengths[connected] += int(round(lengths[l]/2,0))
                    
    # Set to True to show which ligaments are considered to be within a
    # single node. 
    if showSteps:       
        ligaments_small = ligaments > 0
        ligaments_small = ligaments_small.astype('uint8')
        ligaments_small[ligaments_small==1] = 2
        for i in too_small:
            ligaments_small[ligaments==i] = 1
        display.showSkel(ligaments_small, mask, dialate=False,
                title=("Green = within node ligaments, " \
                       "White = between node ligaments"))


    # filter out background and extra small lengths
    lengths[0] = 0
    allMeasurements = lengths[lengths>0]
    lengths=lengths[lengths>estSize*filter_out_mult]

    if len(lengths) == 0:
        if returnAll:
            return 0,0,0
        else:
            return 0,0

   
    # Get a lognormal fit.
    fit = lognorm.fit(lengths.flatten(), floc=0)
    pdf_fitted = lognorm.pdf(lengths.flatten(),
                             fit[0], loc=fit[1], scale=fit[2])

    # Get gaussian fit.
    gfit = norm.fit(lengths.flatten())

    
    # Get average and standard deviation for lognormal fit.
    logaverage = lognorm.mean(fit[0], loc=fit[1], scale=fit[2])
    logstd = lognorm.std(fit[0], loc=fit[1], scale=fit[2])


    # Get average and standard deviation for normal fit.
    average = norm.mean(gfit[0], gfit[1])
    std = norm.std(gfit[0], gfit[1])
    numbMeasured = lengths.size

    if showSteps:
        display.showSkel(ligaments, mask, dialate=False)

    if showSteps:
        display.showHist(lengths, gauss=True, log=True,
                         title='Ligament Lengths',
                         xlabel='Ligament lengths [pixels]')

    if pdf is not None:
        inout.pdfSaveSkel(pdf, ligaments, mask, dialate=True)
        inout.pdfSaveHist(pdf, lengths, gauss=True, log=True,
                       title='Ligament Lengths',
                       xlabel='Ligament lengths [pixels]')
        

    
    if returnAll:
        # Change to return average, std... for mean and SD of normal PDF.
        return logaverage, logstd, allMeasurements
    else:
        return logaverage, logstd

def area(mask, estSize, returnAll=False, showSteps=False):
    """
    Calculates the area of pores or seperated ligaments.
    Only meaningful if the phase is not fully connected.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase. 
    estSize : float64
        The estimated diameter.  Used to set some parameters.
    showSteps : bool, optional
        If set to true, figures will be displayed that shows the results of
        intermediate steps when calculating the diameter.
    returnAll : bool, optional
        If set to true, a list of all the measurements will be returned.

    Returns
    -------
    average : float64
        Average ojbect area.
    SD : float64
        Standard deviation of areas.
    areas: 1D array of float64, optional
        A list of all measured object areas.
    """
    
    # Return 0 if the mask is fully connected.
    if isConnected(mask):
        if returnAll:
            return 0,0,0
        else:
            return 0,0
        
    im = mask.copy()
    
    # Remove small areas that are probably just poking through the other phase.
    small = 3.14 * (estSize/2)**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = remove_small_objects(im, small)
    
    # label each blob.  0 = background, 1 = blobs touching edge.
    lig_labels = label(im, background=0)
    #subtract 1 for the ones connected to edge.
    num_blobs = np.amax(lig_labels) - 1 
    
    areas = np.bincount(lig_labels.flatten()) # Count pixels in each ligament.
    areas[1:][::-1].sort() # Sort the areas (ignore background).
    average = np.average(areas[2:]) # Ignore background and edge blobs.
    std_dev = np.std(areas[2:])

    #If more area of ligament is connected to the background than not,
    #then there is basically 100% in plane connectivity.
    #Report 0 in this case.
    if areas[1] > np.sum(areas[2:]):
        average = 0
        std_dev = 0
                         

    # animate labels
    if showSteps:
        fig = plt.figure()
        ims = []
        ims.append([plt.imshow(im, cmap=plt.get_cmap('gray'), animated=True)])
        ims.append([plt.imshow(lig_labels, animated=True)])
        ani = animation.ArtistAnimation(fig, ims, 1000, True, 1000)
        plt.axis('off')
        plt.ion()
        plt.show()
        while plt.get_fignums():
            try:
                plt.pause(0.1)
            except:
                pass
        plt.ioff()

    if returnAll:
        return average, std_dev, areas[2:]
    else:
        return average, std_dev

def connectedLength(mask, estSize, returnAll=False, showSteps=False, pdf=None):
    """
    Calculates the average sum connected ligament length size and standard
    deviation of the passed of the passed mask.  This is the average length
    of the backbone of the objects in the mask when summing the length of each
    fork and branch in the object.

    Parameters
    ----------
    mask : 2D array of bool
        Mask showing the area of one phase.
    estSize : float64
        The estimated diameter.  Used to set some parameters.
    returnAll : bool, optional
        If set to true, a list of all the measurements will be returned.
    showSteps : bool, optional
        If set to true, figures will be displayed that shows the results of
        intermediate steps when calculating the diameter.
    pdf : matplotlib.backends.backend_pdf.PdfPages, optional
        If provided, information will be output to the pdf file.

    Returns
    -------
    average : float64
        Average sum length.
    SD : float64
        Standard deviation of sum lengths.
    lengths: 1D array of float64, optional
        A list of all measured sum lengths.
    """

    # Adjustable parameters.
    # These have been highly tested and changing these may throw off results.
    smooth_div = 5 # Sets the structure element radius for edge smoothing.
    
    im = mask.copy()
    
    # Return 0 if the mask is fully connected.
    if isConnected(mask):
        if returnAll:
            return 0,0,0
        else:
            return 0,0

    # Do some extra smoothing to the mask.
    smooth_rad = max(estSize/smooth_div, 1)
    im = segment.smoothEdges(im, smooth_rad)

    skel = skeleton(mask, estSize)
    dist = distanceTransform(mask)

    # Label each section.
    labels = label(im, connectivity=2)
    labels[labels==1] = 0 #Remove labels touching the edge
    labels = skel * labels

    # Get lengths.
    lengths = np.bincount(labels[labels>0].flatten())

    # When the skel backbone does not touch the edge of the object, add the
    # distance to the edge.
    # Find number of neighboring pixels.
    neighbors = convolve2d(skel, np.ones((3,3)), mode='same')
    neighbors = neighbors * skel
    term_neighbors = neighbors < 3
    term_neighbors[neighbors == 0] = 0    
    # If the endpoint of a ligament does not touch edge,
    # add the distance to the edge.
    for l in np.unique(labels):
        if l > 1:
            this_label = labels == l
            sum_places = term_neighbors*this_label
            to_sum = sum_places * dist
            lengths[l] += np.sum(to_sum)

    lengths = lengths[lengths>0]
    average = np.average(lengths)
    SD = np.std(lengths)

    if showSteps:
        display.showSkel(labels, mask, dialate=False)

    if showSteps:
        display.showHist(lengths, gauss=True, log=True,
                         title='Sum connected lengths',
                         xlabel='Object lengths [pixels]')
        
    if pdf is not None:
        inout.pdfSaveSkel(pdf, labels, mask, dialate=True)
        inout.pdfSaveHist(pdf, lengths, gauss=True, log=True,
                       title='Sum connected lengths',
                       xlabel='Object lengths [pixels]')
        
    if returnAll:
        return average, SD, lengths
    else:
        return average, SD
