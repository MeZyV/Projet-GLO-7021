import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import RandomAffine as RandomAffine2

from PIL import Image
from kornia.augmentation import RandomAffine


class COCO_unsupervised(Dataset):

    def __init__(self, config, root_dir, transform=None):
        """
        Args:
            config (dict): Essentially for homography
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.root_dir = root_dir
        self.transform = transform
        self.names = np.array(os.listdir(root_dir))
        self.n_homographies = self.config["n_homographies"]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.names[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
        else:
            image = pil_to_tensor(image)

        if self.n_homographies > 0:
            params_homog = self.config['augmentation']['homographic']
            N_image = image.repeat(self.n_homographies, 1, 1, 1)

            N_image, homography, homography_invert = get_homography(N_image, **params_homog)

            return {
                'image': N_image,
                'homography': homography,
                'homography_invert': homography_invert
            }

        else:
            return {
                'image': image
            }


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

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
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

        twin_im, homography, homography_invert = get_homography(image)

        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        return {
            'image': image,
            'landmarks': landmarks,
            'twin_im': twin_im,
            'homography': homography,
            'homography_invert': homography_invert
        }


def get_homography(image: torch.Tensor, degrees=(-5, 5), translate=(0.2, 0.2), scale=(1.2, 1.5), shear=(-15, 15)):
    # TODO: not enough variations
    aug = RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, return_transform=True)
    twin_im, homography = aug(image)
    if len(homography.size()) < 3:
        homography = homography.unsqueeze(0)

    # homography_invert = torch.inverse(homography)
    homography_invert = invert_homography(homography)

    return twin_im, homography, homography_invert


def invert_homography(homography):
    homography_invert = torch.zeros_like(homography)
    homography_invert[:, 2, 2] = 1.0
    homography_invert[:, :2, :2] = torch.transpose(homography[:, :2, :2], dim0=1, dim1=2)
    homography_invert[:, :2, 2] = -homography[:, :2, 2]

    return homography_invert
