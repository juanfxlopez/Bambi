from sk_dataloader_inter_contour import DatasetCells, CellTrainData
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
from skimage import exposure

data = DatasetCells( pd.read_csv("../data/GenData/train_input_ids.csv"),
                pd.read_csv("../data/GenData/train_labels_ids.csv"),
                pd.read_csv("../data/GenData/train_inter_ids.csv"),
                pd.read_csv("../data/GenData/train_contour_ids.csv"))

# Rotations and crops           = 7
'''rotations = [ iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Affine(rotate=90),
    iaa.Affine(rotate=270),
    iaa.CropAndPad(percent=(-0.3, 0.3)),
    iaa.Crop(percent=0.2),
    iaa.Crop(percent=0.4)
 ]

for i in tqdm(range(len(rotations)), total=len(rotations)):
    for j, sample in tqdm(enumerate(data), total=len(data)):
        img, label, inter, contour= sample
        #print(img.shape)
        new_img = rotations[i].augment_image(img)
        new_label = rotations[i].augment_image(label)
        new_inter = rotations[i].augment_image(inter)
        new_contour = rotations[i].augment_image(contour)
        new_img = Image.fromarray(new_img)
        new_label = Image.fromarray(new_label)
        new_inter = Image.fromarray(new_inter)
        new_contour = Image.fromarray(new_contour)
        new_img.save("../data/GenData/TrainData/images/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
        new_label.save("../data/GenData/TrainData/labels/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
        new_inter.save("../data/GenData/TrainData/watershed/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
        new_contour.save("../data/GenData/TrainData/contours/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
print("Finished augmentations for rotations and cropping..")


# Blur                          = 5
blur = [ iaa.GaussianBlur(sigma=0.9),
    iaa.GaussianBlur(sigma=2.9),
    iaa.AverageBlur(k=7),
    iaa.AverageBlur(k=9),
    iaa.MedianBlur(k=7),
]

for i in tqdm(range(len(blur)), total=len(blur)):
    for j, sample in tqdm(enumerate(data), total=len(data)):
        img, label,inter, contour = sample
        new_img = blur[i].augment_image(img)
        new_img = Image.fromarray(new_img)
        label = Image.fromarray(label)
        inter = Image.fromarray(inter)
        contour = Image.fromarray(contour)
        new_img.save("../data/GenData/TrainData/images/" + str("%04d" % j) + "_blur_" + str(i) + "_.png")
        label.save("../data/GenData/TrainData/labels/"+ str("%04d" % j) + "_blur_" + str(i) + "_.png")
        inter.save("../data/GenData/TrainData/watershed/" + str("%04d" % j) + "_blur_" + str(i) + "_.png")
        contour.save("../data/GenData/TrainData/contours/" + str("%04d" % j) + "_blur_" + str(i) + "_.png")
print("Finished augmentations for blurring..")

# Miscelaneous                  = 4
misc = [ iaa.AdditiveGaussianNoise(scale=0.2*255),
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Sharpen(alpha=0.6, lightness=0.75),
    iaa.Sharpen(alpha=1.0, lightness=0.75),
]

for i in tqdm(range(len(misc)), total=len(misc)):
    for j, sample in tqdm(enumerate(data), total=len(data)):
        img, label,inter,contour = sample
        new_img = misc[i].augment_image(img)
        new_img = Image.fromarray(new_img)
        label = Image.fromarray(label)
        inter = Image.fromarray(inter)
        contour = Image.fromarray(contour)  
        new_img.save("../data/GenData/TrainData/images/" + str("%04d" % j) + "_sharp_" + str(i) + "_.png")
        label.save("../data/GenData/TrainData/labels/"+ str("%04d" % j) + "_sharp_" + str(i) + "_.png")
        inter.save("../data/GenData/TrainData/watershed/" + str("%04d" % j) + "_sharp_" + str(i) + "_.png")
        contour.save("../data/GenData/TrainData/contours/" + str("%04d" % j) + "_sharp_" + str(i) + "_.png")
print("Finished augmentations for miscalleneous..")'''

# Adjusting exposure            = 1
print("Adjusting Exposure..")
for j, sample in tqdm(enumerate(data), total=len(data)):
    img, label,inter,contour = sample
    new_img = exposure.adjust_gamma(img, gamma=0.4, gain=0.9)
    new_img = Image.fromarray(new_img)
    label = Image.fromarray(label)
    inter = Image.fromarray(inter)
    contour = Image.fromarray(contour)  
    new_img.save("../data/GenData/TrainData/images/" + str("%04d" % j) + "_exposure_.png")
    label.save("../data/GenData/TrainData/labels/" + str("%04d" % j) + "_exposure_.png")
    inter.save("../data/GenData/TrainData/watershed/" + str("%04d" % j) + "_exposure_.png")
    contour.save("../data/GenData/TrainData/contours/" + str("%04d" % j) + "_exposure_.png")
print("Finished..")
