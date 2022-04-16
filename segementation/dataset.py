"""
Author: lu.zhiping@u.nus.edu

Dataset Class for Road Segmentation
"""
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class DriveableDataset(Dataset):
    """
    Dataset object for Driveable BDD100k
    """
    def __init__(self, file_list: list,
                 image_path: str,
                 mask_path: str,
                 transform=None):
        self.file_list = file_list
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, f"{self.file_list[idx]}.jpg")
        mask_path = os.path.join(self.mask_path, f"{self.file_list[idx]}.png")
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        return image, mask
    