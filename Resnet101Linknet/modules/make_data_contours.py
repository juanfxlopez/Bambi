import os
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage import measure
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing
from skimage.morphology import dilation, watershed, square, erosion
from skimage.measure import label, regionprops
from skimage import io

def create_watershed(labels):
    mask = labels.copy()
    mask[mask > 0] = 1
    dilated = binary_dilation(mask, iterations=10)
    mask_wl = watershed(dilated, labels, mask=dilated, watershed_line=True)
    mask_wl[mask_wl > 0] = 1
    contours = dilated - mask_wl
    contours = binary_dilation(contours, iterations=1)
    return contours

TRAIN_PATH = "../data/DSB-Stage1/"
#TEST_PATH = "../"

train_ids = next(os.walk(TRAIN_PATH))[1]
traind_ids = sorted(train_ids)
print("Getting images and reconstructing masks..")
sys.stdout.flush()

alpha = 0.5
beta= 1.0- alpha

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH+id_
    img = cv2.imread(path + "/images/" + id_ + ".png", cv2.IMREAD_UNCHANGED)
    im = Image.fromarray(img)
    im.save("../data/GenData/TrainData/images/" + str("%04d" % (n + 65)) + "_.png")
    
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
    mask_original = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
    intermask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
    for mask_file in next(os.walk(path+"/masks/"))[2]:
        #mask_ = cv2.imread(path + "/masks/" + mask_file, cv2.IMREAD_UNCHANGED)
        mask_ = cv2.imread(path + "/masks/" + mask_file, cv2.IMREAD_UNCHANGED)
        # Find contours at a constant value of 0.8
        mask_plt = mask_.copy()
        mask2_plt=mask.copy()
        #contours = measure.find_contours(mask_, 0.8)
        contours_,hierarchy_ = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Isolate largest contour
        #contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        #biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        cv2.drawContours(mask_plt,contours_, -1, (255,0,0), 10)
        cv2.drawContours(mask2_plt,contours, -1, (255,0,0), 10)

        #cv2.imshow('mask_plt', mask_plt)
        #cv2.imshow('mask2_plt', mask2_plt)


        intermask_= cv2.bitwise_and(mask2_plt,mask_plt)
        mask = np.maximum(mask, mask_) # adding the new cell to the image
        mask_original= np.maximum(mask_original, mask_)
        mask =mask - intermask_ # extracting the intersection
        retval, mask= cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        #print(maskita.shape)
        intermask = np.maximum(intermask, intermask_)
        #cv2.imshow('intermask',intermask)
       
        #cv2.imshow('mask',mask)
        #cv2.imshow('mask_original',mask_original)
        #dst=cv2.addWeighted(mask, alpha, intermask, beta, 0.0)

        wtrshed = create_watershed(label(mask))
        plt.imsave("watershed.png", wtrshed)
        watershed_ = cv2.imread("watershed.png", cv2.IMREAD_GRAYSCALE)
        retval, watershed_ = cv2.threshold(watershed_, 200, 255, cv2.THRESH_BINARY)
        #cv2.imshow('watershed',watershed_)

        '''contour2 = create_contour(label(mask_original))
        plt.imsave("contours_original.png", contour2)
        contours_original = cv2.imread("contours_original.png", cv2.IMREAD_GRAYSCALE)
        cv2.imshow('contour_original',contours_original)'''

        maskita= mask_original - watershed_
        retval, maskita = cv2.threshold(maskita, 200, 255, cv2.THRESH_BINARY)
        #cv2.imshow('maskita', maskita)

        mask2_plt=np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
        contours2,hierarchy = cv2.findContours(maskita, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #now count cells
        cv2.drawContours(mask2_plt,contours2, -1, (255,0,0), 2)
        #cv2.imshow('plt2', mask2_plt)
  
        
        #dst = cv2.addWeighted(dst, 1,watershed_, 0.6, 0.0)
        #cv2.imshow('dst',dst)
        #cv2.waitKey(0)

    
    mask_label = Image.fromarray(mask_original)
    #mask_labelinter = Image.fromarray(maskita)
    #mask_inter = Image.fromarray(intermask)
    mask_watershed=Image.fromarray(watershed_)
    mask_contours=Image.fromarray(mask2_plt)
    #mask_dst = Image.fromarray(dst)
    
    mask_label.save("../data/GenData/TrainData/labels/" + str("%04d" % (n + 65)) + "_.png")
    #mask_labelinter.save("../data/GenData/TrainData/labels_inter/" + str("%04d" % (n + 65)) + "_.png")
    mask_watershed.save("../data/GenData/TrainData/watershed/" + str("%04d" % (n + 65)) + "_.png")
    #mask_inter.save("../data/GenData/TrainData/intersections/" + str("%04d" % (n + 65)) + "_.png")
    mask_contours.save("../data/GenData/TrainData/contours/" + str("%04d" % (n + 65)) + "_.png")
    #mask_dst.save("../data/GenData/TrainData/overlays/" + str("%04d" % (n + 65)) + "_.png")