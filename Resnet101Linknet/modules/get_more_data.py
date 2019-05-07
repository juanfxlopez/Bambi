import numpy as np
import pandas as pd
from skimage.data import imread
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import os
import errno

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    #print(starts)
    #print(ends)
    #print(lengths)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        #print("-----")
        #print(lo)
        #print(hi)
        #print("-----")
    
    return img.reshape((shape[1],shape[0]))

def create_masks(ImageId,masks_n,train_path):
    images_path = train_path + ImageId + "/images/"
    masks_path= train_path + ImageId + "/masks/"

    

    img = cv2.imread(images_path + ImageId + ".png")
    #print(img.shape)
    #im = Image.fromarray(img)
    #im.save(images_path + ImageId + "_.png")

    #cv2.imshow('im',img)
    

    img_masks = masks_n.loc[masks_n['ImageId'] == ImageId, 'EncodedPixels'].tolist()

    # Take the individual masks and create an array for each one
    exist_mask=False
    all_masks = np.zeros((img.shape[0], img.shape[1]))
    if (img_masks!=['1 1']):
        exist_mask=True

        try:
            if (train_path=="../data/DSB-Stage1-test/"):
                #os.mkdir(folder_path)
                #os.mkdir(images_path)
                os.mkdir(masks_path)
            elif (train_path=="../data/DSB-Stage2/"): 
                os.mkdir("../data/DSB-Stage2-fixed/"+ImageId)
                os.mkdir("../data/DSB-Stage2-fixed/"+ImageId+"/images/")
                os.mkdir("../data/DSB-Stage2-fixed/"+ImageId+"/masks/")

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass


        for n,mask in enumerate(img_masks):
            single_mask = np.zeros((img.shape[0], img.shape[1]))
            #single_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8, order='C')
            # Note that NaN should compare as not equal to itself
            if mask == mask:
                decoded_mask = rle_decode(mask, (img.shape[0], img.shape[1])).T
                #print(single_mask.shape)
                #print(decoded_mask.shape)
                single_mask += decoded_mask
                all_masks += decoded_mask
                #single_mask = single_mask.astype(np.uint8)
                single_mask_im = Image.fromarray(np.uint8(single_mask*255))
                if (train_path=="../data/DSB-Stage1-test/"):
                    single_mask_im.save(masks_path + ImageId +"_"+ str(n) + ".png")
                elif (train_path=="../data/DSB-Stage2/"): 
                    im = Image.fromarray(img)
                    im.save("../data/DSB-Stage2-fixed/" + ImageId + "/images/"  + ImageId + "_.png")
                    single_mask_im.save("../data/DSB-Stage2-fixed/" + ImageId + "/masks/" + ImageId +"_"+ str(n) + ".png")
                #single_mask_im.save(+ ImageId +"_"+ str(n) + ".png")
                #cv2.imshow('single_mask',single_mask)
                #cv2.waitKey(0)
        #cv2.imshow('all_masks',all_masks)
        #cv2.waitKey(0)
    return exist_mask
#######################



masks1 = pd.read_csv('../data/stage1_solution.csv')
num_masks1 = masks1.shape[0]
print('number of training images', num_masks1)
masks1.head()

TRAIN_PATH1 = "../data/DSB-Stage1-test/"
train_ids1 = next(os.walk(TRAIN_PATH1))[1]
traind_ids1 = sorted(train_ids1)
count=0
for ImageId in tqdm(train_ids1, total=len(train_ids1)):
    #print(ImageId)
    exist_mask=create_masks(ImageId,masks1,TRAIN_PATH1)
    if exist_mask:
        count+=1
print(count)

masks2 = pd.read_csv('../data/stage2_solution_final.csv')
num_masks2 = masks2.shape[0]
print('number of training images', num_masks2)
masks2.head()

TRAIN_PATH2 = "../data/DSB-Stage2/"
train_ids2 = next(os.walk(TRAIN_PATH2))[1]
traind_ids2 = sorted(train_ids2)
count=0
for ImageId in tqdm(train_ids2, total=len(train_ids2)):
    #print(ImageId)
    exist_mask = create_masks(ImageId,masks2,TRAIN_PATH2)
    if exist_mask:
        count+=1
print(count)
    
    