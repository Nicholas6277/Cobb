import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class AugmentedImageDataset(Dataset):
    def __init__(self, images, labels, augmentation=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.augmentation = augmentation

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]

        # Read image and label
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # Add a channel dimension
        img = img[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

        # Apply augmentation
        if self.augmentation:
            sampled = self.augmentation(image=img, mask=label)
            img = sampled['image']
            label = sampled['mask']

        return torch.div(img, 255), torch.div(label, 255)

    def __len__(self):
        return len(self.images)



def get_simple_training_augmentation():
    return albu.Compose([
        albu.Flip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        albu.PadIfNeeded(1216, 512, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2(transpose_mask=True)
    ])


def get_complex_training_augmentation():
    return albu.Compose([
        albu.Flip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        albu.OneOf([
            albu.ElasticTransform(p=0.8),
            albu.GridDistortion(p=0.5)
        ], p=0.8),
        albu.OneOf([
            albu.GaussNoise(p=0.9),
            albu.GaussNoise(p=0.6),
        ], p=0.8),
        albu.PadIfNeeded(1216, 512, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        ToTensorV2(transpose_mask=True)
    ])

def get_validation_augmentation():
    return albu.Compose([
        albu.PadIfNeeded(1216, 512, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        ToTensorV2(transpose_mask=True)
    ])
