import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from kornia.augmentation import RandomAffine


class COCO(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, landmark_bool=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_bool = landmark_bool

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = pil_to_tensor(image)

        if self.landmark_bool:
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 3)
        else:
            landmarks = 0

        if self.transform:
            image = self.transform(image)

        twin_im, homography, homography_invert = self.get_twin(image)
        
        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        return {
            'image': image,
            'landmarks': landmarks,
            'twin_im': twin_im,
            'homography': homography,
            'homography_invert': homography_invert
        }

    def get_twin(self, image: torch.Tensor, degrees=(-5, 5), translate=(0.2, 0.2), scale=(1.2, 1.5), shear=(-15, 15)):
        image_tens = image.type(torch.float32)
        aug = RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,  return_transform=True)
        twin_im, homography = aug(image_tens)
        if len(homography.size()) < 3:
            homography = homography.unsqueeze(0)
        homography_invert = torch.inverse(homography)
        return twin_im, homography, homography_invert
