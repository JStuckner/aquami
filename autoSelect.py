#!/usr/bin/env python3
"""
This module is for automatically seperating the image data from the meta data.
It is a work in progress.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from aquami import inout

try:
    from pytesseract import image_to_string
except ImportError:
    pass
    #print('Install pytesseract to enable automatic text recognition')

def profiles():
    profiles = ["Katy - Leo",
                "1",
                "2"
                ]
    return profiles


def getDimentions(profile):
    # returns (start row, end row, start column, end column)
    if profile == "1":
        imageLoc = (0,1768, 0, 2043)
        metaLoc = (1767, 1886, 0, 2043)
        scaleLoc = (1772, 1827, 1261, 2042)
        return imageLoc, metaLoc, scaleLoc
    elif profile == "Katy - Leo":
        imageLoc = (0,690, 0, 1023)
        metaLoc = (691, 766, 0, 1023)
        scaleLoc = (700, 758, 265, 400)
        return imageLoc, metaLoc, scaleLoc
    
def seperate(original, imageLoc, metaLoc, scaleLoc):
    i = imageLoc
    m = metaLoc
    s = scaleLoc
    image = original[i[0]:i[1], i[2]:i[3]]
    meta = original[m[0]:m[1], m[2]:m[3]]
    scale = original[s[0]:s[1], s[2]:s[3]]
    return image, meta, scale

def pixelSize(scale, profile):
    
    if profile == "Katy - Leo":
        # How many pixels long is the scalebar?
        barSizeList = []
##        plt.imshow(scale)
##        plt.show()
        try:
            scale = scale[:,:,0]
        except:
            pass
            
        for i in range(20):
            Bar = scale[25+i, :]
            barSizeList.append(np.count_nonzero(~Bar))
        #print(barSizeList)
        barSize = max(barSizeList)
        scaleNumber = scale[:25, :]


        # Trim right side
        shape = scaleNumber.shape
        for i in range(100):
            if np.count_nonzero(~scaleNumber[:, :2]) == 0:
                scaleNumber = scaleNumber[:, 2:]
        # Trim left side
        for i in range(100):
            if np.count_nonzero(~scaleNumber[:, -2:]) == 0:
                scaleNumber = scaleNumber[:, :-2,]
        scaleNumber = np.pad(scaleNumber, ((0,0),(2,2)), 'maximum')

        try:
            scaleNumber = scaleNumber[:,:]
        except:
            pass

        scaleNumber = Image.fromarray(np.uint8(plt.cm.gist_earth(scaleNumber)*255))

        
        try:
            strScaleNum = image_to_string(scaleNumber)
            print(strScaleNum)
        except:
            strScaleNum = ''

        if strScaleNum == '':
            plt.imshow(scaleNumber)
            plt.title('Failed to read scale bar number.\nClose this figure and input the number manually.')
            plt.axis('off')
            plt.show()
            print('Read ', strScaleNum, '.', sep='')
            intScaleNum = int(input('What is the scale bar number?'))
            strScaleNum = 'nm'
        else:
            intScaleNum = 0
            for c in strScaleNum:
                try:
                    intScaleNum = 10*intScaleNum + int(c)
                except:
                    pass

        if 'nm' not in strScaleNum: #then must be micrometers
            intScaleNum *= 1000

        #print(intScaleNum, barSize)
        pixelSize = round(intScaleNum/barSize, 2)   

        return pixelSize
            
    elif profile == "1":
        # Extract the scale bar data.
        scaleBar = scale[:56, 1262:2040]
        rows,cols = scaleBar.shape


        # How many pixels long is the scalebar?
        leftBar = scaleBar[30, :250]
        rightBar = scaleBar[30, 550:]
        leftZeros = len(leftBar) - np.count_nonzero(leftBar)
        rightZeros = len(rightBar)- np.count_nonzero(rightBar)
        barSize = cols - leftZeros - rightZeros

        # Read the scale bar number
        scaleNumber = scaleBar[5:, 300:500]
##        plt.imshow(scaleNumber)
##        plt.show()
        scaleNumber = scaleNumber < 120
        scaleNumber = inout.uint8(scaleNumber)
        scaleNumber = Image.fromarray(np.uint8(plt.cm.gist_earth(scaleNumber)*255))
        strScaleNum = image_to_string(scaleNumber)
        try:
            strScaleNum = image_to_string(scaleNumber)
        except:
            strScaleNum = ''
            
        if strScaleNum == '':
            plt.imshow(scaleNumber)
            plt.title('Failed to read scale bar number.\nClose this figure and input the number manually.')
            plt.axis('off')
            plt.show()
            print('Read ', strScaleNum, '.', sep='')
            intScaleNum = int(input('What is the scale bar number?'))
            strScaleNum = 'nm'
        else:
            intScaleNum = 0
            for c in strScaleNum:
                try:
                    intScaleNum = 10*intScaleNum + int(c)
                except:
                    pass

        if 'nm' not in strScaleNum: #then must be micrometers
            intScaleNum *= 1000

        pixelSize = round(intScaleNum/barSize, 2)   

        return pixelSize

def autoDetect(im, profile):
    if profile == "Katy - Leo":
        imageLoc, metaLoc, scaleLoc = getDimentions(profile)
        image, meta, scale = seperate(im, imageLoc, metaLoc, scaleLoc)
        pixel = pixelSize(scale, profile=profile)
        return image, pixel
    
    elif profile == "1":
        imageLoc, metaLoc, scaleLoc = getDimentions(profile)
        image, meta, scale = seperate(im, imageLoc, metaLoc, scaleLoc)
##        plt.imshow(image)
##        plt.show()
##        plt.imshow(meta)
##        plt.show()
        pixel = pixelSize(meta, profile)
        return image, pixel
    
