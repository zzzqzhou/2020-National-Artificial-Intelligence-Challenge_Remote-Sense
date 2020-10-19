import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ReSe(Dataset):
    CLASSES_LABEL = [100, 200, 300, 400, 500, 600, 700, 800]
    def __init__(self, base_dir=None, split=None, num=None, transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(os.path.join(self.base_dir, 'train.list'), 'r') as f:
                self.image_list = f.readlines()
        elif self.split == 'val':
            with open(os.path.join(self.base_dir, 'val.list'), 'r') as f:
                self.image_list = f.readlines()
        elif self.split == 'total_train':
            with open(os.path.join(self.base_dir, 'total_train.list'), 'r') as f:
                self.image_list = f.readlines()
        elif self.split == 'test':
            with open(os.path.join(self.base_dir, 'test.list'), 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image_name = self.image_list[item]

        if self.split == 'test':
            img = Image.open(os.path.join(self.base_dir, image_name))
            if self.transform:
                img = self.transform(img)
            if 'tif' in image_name:
                return img, image_name.replace('.tif', '')
            elif 'jpg' in image_name:
                return img, image_name.replace('.jpg', '')
        
        img = Image.open(os.path.join(self.base_dir, 'image', image_name))
        if 'tif' in image_name:
            mask = Image.open(os.path.join(self.base_dir, 'label', image_name.replace('tif', 'png')))
        elif 'jpg' in image_name:
            mask = Image.open(os.path.join(self.base_dir, 'label', image_name.replace('jpg', 'png')))
        mask = mask2label(mask)
        if 'train' in self.split:
            rand = random.random()
            if rand < 1 / 6:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif rand < 2 / 6:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            elif rand < 3 / 6:
                img = img.transpose(Image.ROTATE_90)
                mask = mask.transpose(Image.ROTATE_90)
            elif rand < 4 / 6:
                img = img.transpose(Image.ROTATE_180)
                mask = mask.transpose(Image.ROTATE_180)
            elif rand < 5 / 6:
                img = img.transpose(Image.ROTATE_270)
                mask = mask.transpose(Image.ROTATE_270)
        img = self.transform(img)
        mask = torch.from_numpy(np.array(mask).astype(np.uint8)).long()
        if self.split == 'val':
            return img, mask, image_name.replace('.tif', '') 
        return img, mask


def mask2label(slice):
    slice = np.array(slice).astype(np.uint16)
    slice = slice // 100 - 1
    slice = Image.fromarray(slice.astype(np.uint8), mode='P')
    return slice