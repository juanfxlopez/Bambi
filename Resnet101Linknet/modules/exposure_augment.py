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


print("Adjusting Exposure..")
for j, sample in tqdm(enumerate(data), total=len(data)):
    img, label,inter = sample
    new_img = exposure.adjust_gamma(img, gamma=0.4, gain=0.9)
    new_img = Image.fromarray(new_img)
    label = Image.fromarray(label)
    inter = Image.fromarray(inter)
    new_img.save("../data/GenData/TrainData/images/" + str("%04d" % j) + "_exposure_.png")
    label.save("../data/GenData/TrainData/labels/" + str("%04d" % j) + "_exposure_.png")
    inter.save("../data/GenData/TrainData/watershed/" + str("%04d" % j) + "_exposure_.png")
print("Finished..")