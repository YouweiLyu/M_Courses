import os
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision
import cv2
import torch
import utils

class MnistDataSet(data.Dataset):
    def __init__(self, path, split):
        super(MnistDataSet, self).__init__()
        self.split = split
        if self.split == 'train' or 'val':
            image_path = os.path.join(path, 'train-images.idx3-ubyte')
            label_path = os.path.join(path, 'train-labels.idx1-ubyte')
        if self.split == 'test':
            image_path = os.path.join(path, 't10k-images.idx3-ubyte')
            label_path = os.path.join(path, 't10k-labels.idx1-ubyte')
        self.image_file = utils.decode_image_file(image_path)/255.
        self.label_file = utils.decode_label_file(label_path)
            
    def __len__(self):
        return self.label_file.shape[0]

    def __getitem__(self, index):
        '''load the datas'''
        image = self.image_file[index][None, :, :]
        label = self.label_file[index]
        return image, label

def dataloader_builder(args):
    trn_set = MnistDataSet(args.data_path, 'train')
    val_set = MnistDataSet(args.data_path, 'test')
    train_loader = torch.utils.data.DataLoader(trn_set, batch_size=args.trn_batch,
        num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(val_set , batch_size=args.val_batch,
        num_workers=args.num_workers, pin_memory=True, shuffle=False)
    return train_loader, test_loader
