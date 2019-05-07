from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io

class DatasetCells(Dataset):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self,file_data_idx, file_label_idx, file_inter_idx, file_contour_idx, transform=None, mode="train"):
        self.data_root = "../data/GenData/"
        #self.data = pd.read_csv(file_path)
        self.file_data_idx = file_data_idx
        self.file_label_idx = file_label_idx
        self.file_inter_idx = file_inter_idx
        self.file_inter_idx = file_contour_idx
        self.transform = transform
        self.mode = mode
        if self.mode is "train":
          self.data_dir = os.path.join(self.data_root, "TrainData/images/")
          self.label_dir = os.path.join(self.data_root, "TrainData/labels/")
          self.inter_dir = os.path.join(self.data_root, "TrainData/watershed/")
          self.contour_dir = os.path.join(self.data_root, "TrainData/labels_inter_inv/")
        elif self.mode is "validation":
            pass
        elif self.mode is "test":
            pass

    def __len__(self):
        return len(self.file_data_idx)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        #image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))
        #label = self.data.iloc[index, 0]
        #train_id = self.file_data_idx["ids"].iloc[index]
        #label_id = self.file_data_idx["ids"].iloc[index]
        if self.mode is "train":
          file_id = self.file_data_idx["ids"].iloc[index]
          self.image_path = os.path.join(self.data_dir, file_id)
          self.label_path = os.path.join(self.label_dir, file_id)# file id should be from label data, buit in this case is the same.
          self.inter_path = os.path.join(self.inter_dir, file_id)
          self.contour_path = os.path.join(self.contour_dir, file_id)
          #image = io.imread(self.image_path)
          #label = io.imread(self.label_path)
          #inter = io.imread(self.inter_path)
          #contour = io.imread(self.contour_path)
          image = Image.open(self.image_path).convert("L")# L -> grayscale channel
          label = Image.open(self.label_path)
          inter = Image.open(self.inter_path)
          contour = Image.open(self.contour_path)

          
          if self.transform is not None:
              image = self.transform(image)
              label = self.transform(label)
              inter = self.transform(inter)   
              contour = self.transform(contour)      
          return image, label, inter, contour
        if self.mode is "validation":
            pass
        if self.mode is "test":
            pass

    def __add__(self, other):
        return ConcatDataset([self, other])


def CellTrainData(data_transform=None, mode="train"):
    file_idxs = pd.read_csv("../data/GenData/train_input_ids.csv")
    label_idxs = pd.read_csv("../data/GenData/train_labels_ids.csv")
    inter_idxs = pd.read_csv("../data/GenData/train_inter_ids.csv")
    contour_idxs = pd.read_csv("../data/GenData/train_contour_ids.csv")
    dataset = DatasetCells(file_idxs, label_idxs, inter_idx, contour_idxs, transform=data_transform, mode=mode)
    return dataset

def CellDataLoader(data_transform=None, mode="train", batch_sz=2, workers=1):
    file_idxs = pd.read_csv("../data/GenData/train_input_ids.csv")
    label_idxs = pd.read_csv("../data/GenData/train_labels_ids.csv")
    inter_idxs = pd.read_csv("../data/GenData/train_inter_ids.csv")
    contour_idxs = pd.read_csv("../data/GenData/train_contour_ids.csv")
    if data_transform is None:
        data_transform = transforms.ToTensor()
    dataset = DatasetCells(file_idxs, label_idxs, inter_idxs, contour_idxs, transform=data_transform, mode="train")
    dataloader = DataLoader(dataset, batch_size=batch_sz, num_workers=workers, shuffle=True)
    return dataloader

def CellTrainValidLoader(data_transform=None, validation_split=0.1, mode="train", batch_sz=2, workers=1):
    file_idxs = pd.read_csv("../data/GenData/train_input_ids.csv")
    label_idxs = pd.read_csv("../data/GenData/train_labels_ids.csv")
    inter_idxs = pd.read_csv("../data/GenData/train_inter_ids.csv")
    contour_idxs = pd.read_csv("../data/GenData/train_contour_ids.csv")
    if data_transform is None:
        data_transform = transforms.ToTensor()
    dataset = DatasetCells(file_idxs, label_idxs,inter_idxs,contour_idxs, transform=data_transform, mode="train")
    
    shuffle_dataset = True
    random_seed = 1234

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_sz, sampler = train_sampler, num_workers=workers)
    validation_loader = DataLoader(dataset, batch_size=batch_sz, sampler = valid_sampler, num_workers=workers)
    
    return train_loader, validation_loader
