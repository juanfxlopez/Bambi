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
import random

from get_data_ids import get_ids_in_list

def create_watershed(labels):
    mask = labels.copy()
    mask[mask > 0] = 1
    dilated = binary_dilation(mask, iterations=500)
    mask_wl = watershed(dilated, labels, mask=dilated, watershed_line=True)
    mask_wl[mask_wl > 0] = 1
    contours = dilated - mask_wl
    contours = binary_dilation(contours, iterations=4)#THIS WAS 2 BEFORE
    return contours

TRAIN_PATH = "../data/Datasets/Kaggle2018_fixed/"
#TEST_PATH = "../"
train_ids = get_ids_in_list(TRAIN_PATH)


#train_ids = next(os.walk(TRAIN_PATH))[1]
#traind_ids = sorted(train_ids)
print("Getting images and reconstructing masks..")
sys.stdout.flush()

alpha = 0.5
beta= 1.0- alpha

for n, img_id in tqdm(enumerate(train_ids), total=len(train_ids)):
    #print("hola")

    if (img_id.endswith("_mask.png")):
        id_ = img_id[:-9]
        img = cv2.imread(TRAIN_PATH + id_ + ".png") # in BGR by default
        img=cv2.resize(img,(1024,1024))
        im = Image.fromarray(img)
        im.save("../data/GenData/TrainData/images/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")

        #mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
        #mask_original = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
        intermask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')

        mask_original = cv2.imread(TRAIN_PATH + img_id, cv2.IMREAD_GRAYSCALE)
        print(mask_original.shape)
        mask_original = cv2.resize(mask_original,(1024,1024))
        retval, mask_original= cv2.threshold(mask_original, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3,3), np.uint8)
        mask= cv2.erode(mask_original, kernel, iterations=1)# THIS WAS ITERATION 1 BEFORE
        retval, mask= cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        wtrshed = create_watershed(label(mask))
        print(wtrshed.shape)
        plt.imsave("watershed.png", wtrshed)
        watershed_ = cv2.imread("watershed.png", cv2.IMREAD_GRAYSCALE)
        retval, watershed_ = cv2.threshold(watershed_, 100, 255, cv2.THRESH_BINARY)
        
        #cv2.imshow('watershed',watershed_)


        #mask_original = cv2.resize(mask_original,(512,512))
        #retval, mask_original= cv2.threshold(mask_original, 10, 255, cv2.THRESH_BINARY)

        #cv2.imshow('mask',mask)
        #cv2.imshow('mask_original',mask_original)

        #dst=cv2.addWeighted(mask_original, alpha, intermask, beta, 0.0)
        dst=mask_original

        
        #mask= cv2.resize(mask,(512,512))
        #watershed_= cv2.resize(watershed_,(512,512))
        #retval, watershed_ = cv2.threshold(watershed_, 10, 255, cv2.THRESH_BINARY)


        maskita= mask_original - watershed_
        retval, maskita = cv2.threshold(maskita, 200, 255, cv2.THRESH_BINARY)
        #cv2.imshow('maskita', maskita)

        mask2_plt=np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
        contours2,hierarchy = cv2.findContours(maskita, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #now count cells
        cv2.drawContours(mask2_plt,contours2, -1, (255,0,0), 5)
        #cv2.imshow('plt2', mask2_plt)
    
        
        dst = cv2.addWeighted(dst, 1,watershed_, 0.6, 0.0)
        #cv2.imshow('dst',dst)
        #cv2.waitKey(0)

        mask2_plt=mask2_plt-maskita
        retval, mask2_plt = cv2.threshold(mask2_plt, 200, 255, cv2.THRESH_BINARY)

        mask3_plt=np.zeros((img.shape[0], img.shape[1],3), dtype=np.uint8, order='C')
        for count,value in enumerate(contours2):
                random_color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                cv2.drawContours(mask3_plt, [value], 0, random_color, 1)
        cellCount = len(contours2)# count + 1
        print(cellCount)
        ret, maskcon = cv2.threshold(mask3_plt, 1, 255, cv2.THRESH_BINARY)
        maskcon= cv2.bitwise_not(maskcon,cv2.THRESH_BINARY)
        conImg_plt = cv2.bitwise_and(img,maskcon) # remove the contours form the original image, so put it to black pixels.

        conImg_plt= cv2.add(mask3_plt,conImg_plt)
        #maskk=cv2.erode(mask, kernel, iterations=1)
        
        mask_label = Image.fromarray(mask_original)
        mask_labelinter = Image.fromarray(maskita)
        mask_inter = Image.fromarray(mask)
        mask_watershed=Image.fromarray(watershed_)
        mask_contours=Image.fromarray(mask2_plt)
        mask_contourscolor=Image.fromarray(conImg_plt)
        mask_dst = Image.fromarray(dst)
        
        mask_label.save("../data/GenData/TrainData/labels/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")
        mask_labelinter.save("../data/GenData/TrainData/labels_inter/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")
        mask_watershed.save("../data/GenData/TrainData/watershed/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")
        mask_inter.save("../data/GenData/TrainData/intersections/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")
        mask_contours.save("../data/GenData/TrainData/contours/" + str("%04d" % ((n-1)/2+ 0)) + "_.png")
        mask_contourscolor.save("../data/GenData/TrainData/contourscolor/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")
        mask_dst.save("../data/GenData/TrainData/overlays/" + str("%04d" % ((n-1)/2 + 0)) + "_.png")
