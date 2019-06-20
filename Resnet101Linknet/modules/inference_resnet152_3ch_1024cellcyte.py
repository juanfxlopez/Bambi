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
#from skimage.filters import threshold_otsu, threshold_adaptive

from helper import load_model
from resnet152inter_linknet_model import ResNetLinkModel
from get_data_ids import get_ids_in_list


start_time=time.time()

segm_model=ResNetLinkModel(input_channels=1,num_classes=3)
#segm_model=nn.DataParallel(segm_model) #This is for multiGPU -> cloud
segm_model=load_model(segm_model, model_dir="./ResNet152inter_linknet_i1024_e12_b8_w1_resized_3ch_intercloud48-09-47_mod.pt")

img_size1=1024
img_size2=1024
trf = transforms.Compose([ transforms.Resize(size=(img_size1, img_size2)), transforms.ToTensor() ])


data_path = "../data/Datasets/CellcyteUV10X_fixed/"
prediction_path = "../test/gr_predictionsinter/"
images = get_ids_in_list(data_path)

thrs=0.26
upper=1
lower=0

for img_id in tqdm(images, total=len(images)):
    if (img_id.endswith("_mask.png")):
        thrs=0.26
    else:
        img = Image.open(data_path + img_id)
        #area = (0, 0, 1024, 1024)
        #img = img.crop(area)
        img1 = img.resize(size=(img_size1, img_size2))
        img = img.convert("L")
        img_in = trf(img)
        img.save(prediction_path+img_id)
        img_in = img_in.unsqueeze(dim=0)
        output = segm_model(img_in)
        output1= output[:,0,:,:]
        output2= output[:,1,:,:]
        output3= output[:,2,:,:]
        pred = sigmoid(output)
        pred = pred.squeeze()
        output_np = pred.detach().cpu().numpy()
        pred1 = sigmoid(output1)
        pred1 = pred1.squeeze()
        output_np1 = pred1.detach().cpu().numpy()
        pred2 = sigmoid(output2)
        pred2 = pred2.squeeze()
        output_np2 = pred2.detach().cpu().numpy()
        pred3 = sigmoid(output3)
        pred3 = pred3.squeeze()
        output_np3 = pred3.detach().cpu().numpy()
        #global_thresh = threshold_otsu(output_np)
        binary_out1 = output_np1 > global_thresh


        binary_out2 = np.where(output_np2 > thrs, upper, lower)
        binary_out3 = np.where(output_np3 > thrs, upper, lower)
        #binary_out = output_np
        #mask = Image.fromarray(binary_out)
        #mask.save(img_id + "_mask.png")
        #plt.imshow(img_id)
        substraction = binary_out1- binary_out2
        substraction = np.where(substraction>0.5,1,0)
        binary_out1 = Image.fromarray(np.uint8(binary_out1*255))
        #binary_out1.save(prediction_path + img_id[0:-4] + "mask.png")
        binary_out2 = Image.fromarray(np.uint8(binary_out2*255))
        #binary_out2.save(prediction_path + img_id[0:-4] + "maskwater.png")
        binary_out3 = Image.fromarray(np.uint8(binary_out3*255))
        #binary_out3.save(prediction_path + img_id[0:-4] + "maskcontour.png")

        substraction = Image.fromarray(np.uint8(substraction*255))
        substraction  = substraction.resize(size=(2448, 2048))
        substraction.save(prediction_path + img_id[0:-4] + "masksubs.png")
        

        #plt.imsave(prediction_path + img_id + "_mask.png", binary_out1)
        #plt.imsave(prediction_path + img_id + "_maskwater.png", binary_out2)
        #plt.imsave(prediction_path + img_id + "_masksubs.png", substraction)

        #plt.imshow(binary_out)
        #plt.title(img_id)
        #plt.show()
