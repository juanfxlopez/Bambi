import numpy as np
from matplotlib import pyplot as plt
from skimage import io, measure, feature, filters
from PIL import Image
import cv2
import random

def cellCount_DrawContours(origImgFilename, procImgFilename, savedImgFilename):

    #origImg = cv2.imread(origImgFilename, cv2.IMREAD_UNCHANGED)
    origImg = cv2.imread(origImgFilename) ##BGR BY DEFAULT (3 CHANNELS)
    #origImg = io.imread(origImgFilename)
    print(origImg.shape)
    #processedImg = io.imread(procImgFilename)
    processedImg = cv2.imread(procImgFilename, cv2.IMREAD_UNCHANGED)
    print(processedImg.shape)
    #img = filters.gaussian(processedImg, 3)
    img=processedImg
    thres = filters.threshold_otsu(img)
    binImg = img > thres

    #binImg = filters.gaussian(binImg, 3)

    nonZero = np.count_nonzero(binImg)
    totalPixels = binImg.shape[0] * binImg.shape[0]
    ratio = nonZero / totalPixels

    print(binImg.shape)

    #contours = measure.find_contours(binImg, 0.1)
    binImg_plt = np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8, order='C')
    #binImg_plt = processedImg.copy()
    contours_,hierarchy_ = cv2.findContours(processedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours_)
    for count,value in enumerate(contours_):
        #cnt = contours_[i]
        random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.drawContours(binImg_plt, [value], 0, random_color, 1)
        #cv2.drawContours(binImg_plt,contours_, 4, (255,0,0), 3)
    cellCount = len(contours_)# count + 1
    ret, mask = cv2.threshold(binImg_plt, 1, 255, cv2.THRESH_BINARY)
    mask= cv2.bitwise_not(mask,cv2.THRESH_BINARY)
    img1_bg = cv2.bitwise_and(origImg,mask) # remove the contours form the original image, so put it to black pixels.

    conImg_plt= cv2.add(binImg_plt,img1_bg)
    #binImg_im = Image.fromarray(binImg_plt)
    #binImg_im.save(savedImgFilename[0:-4] + "contours.png")
    conImg_im = Image.fromarray(conImg_plt)
    conImg_im.save(savedImgFilename)

    '''_, ax = plt.subplots()
    ax.imshow(origImg, interpolation="nearest", cmap=plt.cm.gray)

    for _, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    
    ax.axis("image")
    ax.set_xticks([])
    ax.set_yticks([])'''
    #plt.savefig(savedImgFilename, dpi=200, bbox_inches="tight", pad_inches=0)

    return cellCount, ratio