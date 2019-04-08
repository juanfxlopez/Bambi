import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import sigmoid
from torchvision import transforms
from torchvision import utils
from torch import nn

import time
from PIL import Image
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_adaptive

from helper import load_model
from resnet101inter_linknet_model import ResNetLinkModel
from get_data_ids import get_ids_in_list


start_time=time.time()

segm_model=ResNetLinkModel(input_channels=1,num_classes=2)
#segm_model=nn.DataParallel(segm_model)
segm_model=load_model(segm_model, model_dir="./ResNet101inter_linknet_50.pt")

img_size=256
trf = transforms.Compose([ transforms.Resize(size=(img_size, img_size)), transforms.ToTensor() ])

data_path = "../test/images/"
prediction_path = "../test/gr_predictionsinter/"
images = get_ids_in_list(data_path)

thrs=0.26
upper=1
lower=0

for img_id in tqdm(images, total=len(images)):
    img1 = Image.open(data_path + img_id)
    img1 = img1.resize(size=(img_size, img_size))
    img = Image.open(data_path + img_id).convert("L")
    img_in = trf(img)
    img1.save(prediction_path+img_id)
    img_in = img_in.unsqueeze(dim=0)
    output = segm_model(img_in)
    output1= output[:,0,:,:]
    output2= output[:,1,:,:]
    pred = sigmoid(output)
    pred = pred.squeeze()
    output_np = pred.detach().numpy()
    pred1 = sigmoid(output1)
    pred1 = pred1.squeeze()
    output_np1 = pred1.detach().numpy()
    pred2 = sigmoid(output2)
    pred2 = pred2.squeeze()
    output_np2 = pred2.detach().numpy()
    global_thresh = threshold_otsu(output_np)
    binary_out1 = output_np1 > global_thresh


    binary_out2 = np.where(output_np2 > thrs, upper, lower)
    #binary_out = output_np
    #mask = Image.fromarray(binary_out)
    #mask.save(img_id + "_mask.png")
    #print(binary_out[0])
    #plt.imshow(img_id)
    substraction = binary_out1- binary_out2
    print(output2.shape)
    plt.imsave(prediction_path + img_id + "_mask.png", binary_out1)
    plt.imsave(prediction_path + img_id + "_maskwater.png", binary_out2)
    plt.imsave(prediction_path + img_id + "_masksubs.png", substraction)
    #plt.imshow(binary_out)543
    #plt.title(img_id)
    #plt.show()
