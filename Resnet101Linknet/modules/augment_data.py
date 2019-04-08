from sk_dataloader_inter import DatasetCells, CellTrainData
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
from skimage import exposure

data = DatasetCells( pd.read_csv("../data/GenData/train_input_ids.csv"),
                pd.read_csv("../data/GenData/train_labels_ids.csv"),
                pd.read_csv("../data/GenData/train_inter_ids.csv"))

# Rotations and crops           = 7
rotations = [ iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Affine(rotate=90),
    iaa.Affine(rotate=270),
    iaa.CropAndPad(percent=(-0.3, 0.3)),
    iaa.Crop(percent=0.2),
    iaa.Crop(percent=0.4)
 ]

for i in tqdm(range(len(rotations)), total=len(rotations)):
    for j, sample in tqdm(enumerate(data), total=len(data)):
        img, label, inter = sample
        #print(img.shape)
        new_img = rotations[i].augment_image(img)
        new_label = rotations[i].augment_image(label)
        new_inter = rotations[i].augment_image(inter)
        new_img = Image.fromarray(new_img)
        new_label = Image.fromarray(new_label)
        new_inter = Image.fromarray(new_inter)
        new_img.save("../data/GenData/TrainData/images/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
        new_label.save("../data/GenData/TrainData/labels/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
        new_inter.save("../data/GenData/TrainData/watershed/" + str("%04d" % j) + "_flip_" + str(i) + "_.png")
print("Finished augmentations for rotations and cropping..")
