import torch 
from torch import nn
from torch import optim
from torchvision import transforms

from resnet101inter_linknet_model import ResNetLinkModel
from helper import jaccard, dice, save_model, save_checkpoint, load_checkpoint
from dataloader_inter_contour import DatasetCells, CellTrainValidLoader
#import encoding
from parallel import DataParallelModel, DataParallelCriterion

import time
import copy
from tqdm import tqdm
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--lastepoch", type=int, default=0,
    help="use this if you need to continue training a model")
args = vars(ap.parse_args())

use_cuda = torch.cuda.is_available()
# Hyperparameters
batch_size = 2
nr_epochs = 250
momentum = 0.95
lr_rate = 0.03
milestones = [5, 7, 8, 10, 12, 14, 16, 17, 18]
img_size = 384
gamma = 0.5

#use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#use_cuda = False

segm_model = ResNetLinkModel(input_channels=1, pretrained=True, num_classes=3)

if torch.cuda.device_count() > 1:
     #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
     #segm_model = nn.DataParallel(segm_model)
     #segm_model = encoding.parallel.DataParallelModel(segm_model, device_ids=[0,1,2,3,4,5,6,7])
    segm_model = DataParallelModel(segm_model)
print("Let's use", torch.cuda.device_count(), "GPUs!")
segm_model.to(device)


'''if use_cuda:
    segm_model.cuda()
seg_model = nn.DataParallel(seg_model)'''

mul_transf = [ transforms.Resize(size=(img_size, img_size)), transforms.ToTensor() ]
#optimizer = optim.SGD(segm_model.parameters(), lr=lr_rate, momentum=momentum)
optimizer= optim.Adam(segm_model.parameters(), lr = 0.0001)
#criterion = nn.BCEWithLogitsLoss().cuda() if use_cuda else nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss()
criterion = DataParallelCriterion(criterion)
criterion.to(device)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

train_loader, valid_loader = CellTrainValidLoader(data_transform=transforms.Compose(mul_transf), batch_sz=batch_size, workers=2)

dict_loaders = {"train":train_loader, "valid":valid_loader}


def train_model(cust_model, dataloaders, criterion, optimizer, num_epochs, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    val_acc_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(cust_model)
    best_optimizer_wts = optim.Adam(best_model_wts.parameters(), lr = 0.0001)
    best_optimizer_wts.load_state_dict(optimizer.state_dict())
    start_epoch = args["lastepoch"]+1
    if (start_epoch >1):
        filepath= "./checkpoint_epoch" + str(args["lastepoch"]) + ".pth"
        #filepath="ResNet34watershedplus_linknet_50.pt"
        cust_model, optimizer = load_checkpoint(cust_model,filepath)
        #cust_model = load_model(cust_model,filepath)
    for epoch in range(start_epoch-1,num_epochs,1):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("_"*15)
        for phase in ["train", "valid"]:
            if phase == "train":
                cust_model.train()
            if phase == "valid":
                cust_model.eval()
            running_loss = 0.0
            jaccard_acc = 0.0
            jaccard_acc_inter = 0.0
            jaccard_acc_contour = 0.0
            dice_loss = 0.0

            for input_img, labels, inter, contours in tqdm(dataloaders[phase], total=len(dataloaders[phase])):
                #input_img = input_img.cuda() if use_cuda else input_img
                #labels = labels.cuda() if use_cuda else labels
                #inter = inter.cuda() if use_cuda else inter
                input_img = input_img.to(device)
                labels = labels.to(device)
                inter = inter.to(device)
                contours = contours.to(device)
                label_true=torch.cat([labels,inter,contours], 1)
                #label_true=labels
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    out = cust_model(input_img)
                    #preds = torch.sigmoid(out) 
                    preds=out
                    #print(preds.shape)
                    loss = criterion(preds, label_true)
                    loss = loss.mean()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input_img.size(0)
                #print(labels.shape)
                #preds=torch.FloatTensor(preds)
                #print(preds)
                #preds=torch.cat(preds) #more multiGPU
                #print(preds.shape)
                
                jaccard_acc += jaccard(labels.to(device), torch.sigmoid(preds.to(device))) # THIS IS THE ONE THAT STILL IS ACCUMULATION IN ONLY ONE GPU
                #jaccard_acc_inter += jaccard(inter.to('cpu'), torch.sigmoid(preds.to('cpu')))
                #jaccard_acc_contour += jaccard(contours.to('cpu'), torch.sigmoid(preds.to('cpu')))

                #dice_acc += dice(labels, preds)
            
            epoch_loss = running_loss / len(dataloaders[phase])
            print("| {} Loss: {:.4f} |".format(phase, epoch_loss))
            aver_jaccard = jaccard_acc / len(dataloaders[phase])
            #aver_jaccard_inter = jaccard_acc_inter / len(dataloaders[phase])
            #aver_jaccard_contour = jaccard_acc_contour / len(dataloaders[phase])
            #aver_dice = dice_acc / len(dataloaders[phase])
            print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} | ".format(phase, epoch_loss, aver_jaccard))
            #print("| {} Loss: {:.4f} | Jaccard Average Acc: {:.4f} | Jaccard Average Acc inter: {:.4f}  | Jaccard Average Acc contour: {:.4f}| ".format(phase, epoch_loss, aver_jaccard, aver_jaccard_inter, aver_jaccard_contour))
            print("_"*15)
            if phase == "valid" and aver_jaccard > best_acc:
                best_acc = aver_jaccard
                best_acc_inter = aver_jaccard ## aver_jaccard_inter
                best_epoch_loss = epoch_loss
                #best_model_wts = copy.deepcopy(cust_model.state_dict)
                best_model_wts = copy.deepcopy(cust_model)
                best_optimizer_wts = optim.Adam(best_model_wts.parameters(), lr = 0.0001)
                best_optimizer_wts.load_state_dict(optimizer.state_dict())
            if phase == "valid":
                val_acc_history.append(aver_jaccard)
        print("^"*15)
        save_checkpoint(best_model_wts,best_optimizer_wts,epoch+1,best_epoch_loss,best_acc,best_acc_inter)
        print(" ")
        scheduler.step()
    time_elapsed = time.time() - start_time
    print("Training Complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    #print("Best Validation Accuracy: {:.4f}".format(best_acc))
    #este no#best_model_wts = copy.deepcopy(cust_model.state_dict())
    cust_model.load_state_dict(best_model_wts.state_dict())
    return cust_model, val_acc_history

segm_model, acc = train_model(segm_model, dict_loaders, criterion, optimizer, nr_epochs, scheduler=scheduler)
save_model(segm_model, name="ResNet101inter_linknet_i384_e250_w2_c2_3ch_local.pt")
