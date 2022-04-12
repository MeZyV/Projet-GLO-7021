import os
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset

# Download data from https://www.kaggle.com/competitions/image-matching-challenge-2022/data
# Extract content of the archive to ./data/ImageMatching
class ImageMatching(Dataset):

    def __init__(self, path, verbose=True, transform=None, subset=None):
        self.transform = transform
        self.path = path
        self.subsets = os.listdir(self.path)
        self.subsets.remove('scaling_factors.csv')
        self.scaling_factors = pd.read_csv(os.path.join(self.path, 'scaling_factors.csv'))
        if verbose:
            print(f'Set a subset to laod. By calling set_subset(). Default is "{self.subset}"')
            print('Available subsets:')
            for subset in self.subsets:
                print('\t', subset)
        if subset:
            self.set_subset(subset)
        else:
            self.set_subset(self.subsets[0])

    def set_subset(self, subset):
        assert subset in self.subsets, 'Subset {subset} not found in {self.subsets}'
        self.subset = subset
        self.images = os.listdir(os.path.join(self.path, self.subset, 'images'))
        self.calibration = pd.read_csv(os.path.join(self.path, self.subset, 'calibration.csv'), index_col='image_id')
        self.pair_covisibility = pd.read_csv(os.path.join(self.path, self.subset, 'pair_covisibility.csv'))
        self.scale_factor = float(self.scaling_factors[self.scaling_factors.scene == self.subset]['scaling_factor'])

    def _df_line_to_tensor(self, line, shape):
        mat = torch.tensor([float(x) for x in line.split(' ')])
        return mat.reshape(shape)

    def _compute_intrinsic_matrix(self, calibration_matrix):
        intrinsic_matrix = torch.eye(4)
        intrinsic_matrix[:3, :3] = calibration_matrix
        return intrinsic_matrix
    
    def _compute_extrinsic_matrix(self, rotation_matrix, translation_vector):
        extrinsic_matrix = torch.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_vector
        return extrinsic_matrix

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.subset, 'images', f'{idx}.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        
        calibration_matrix = self._df_line_to_tensor(self.calibration.camera_intrinsics[idx], (3, 3))
        rotation_matrix = self._df_line_to_tensor(self.calibration.rotation_matrix[idx], (3, 3))
        translation_vector = self._df_line_to_tensor(self.calibration.translation_vector[idx], (1, 3))
        
        intrinsic_matrix = self._compute_intrinsic_matrix(calibration_matrix)
        extrinsic_matrix = self._compute_extrinsic_matrix(rotation_matrix, translation_vector)
        
        return {
            'image': image,
            'intrinsic_matrix': intrinsic_matrix,
            'extrinsic_matrix': extrinsic_matrix,
        }


class ImageCovisibility(ImageMatching):

    def __init__(self, path, verbose=True, transform=None, subset=None):
        super(ImageCovisibility, self).__init__(path, verbose, transform, subset)

    def __getitem__(self, idx):
        idx1, idx2 = tuple(self.pair_covisibility.pair[idx].split('-'))
        sample1 = super().__getitem__(idx1)
        sample2 = super().__getitem__(idx2)
        
        image1 = sample1['image']
        image2 = sample2['image']
        intrinsic_matrix1 = sample1['intrinsic_matrix']
        intrinsic_matrix2 = sample2['intrinsic_matrix']
        extrinsic_matrix1 = sample1['extrinsic_matrix']
        extrinsic_matrix2 = sample2['extrinsic_matrix']
        
        covisibility = self.pair_covisibility.covisibility[idx]
        fundamental_matrix = self._df_line_to_tensor(self.pair_covisibility.fundamental_matrix[idx], (3, 3))
        
        return {
            'image1': image1,
            'image2': image2,
            
            'intrinsic_matrix1': intrinsic_matrix1,
            'intrinsic_matrix2': intrinsic_matrix2,
            
            'extrinsic_matrix1': extrinsic_matrix1,
            'extrinsic_matrix2': extrinsic_matrix2,
            
            'covisibility': covisibility,
            'fundamental_matrix': fundamental_matrix,
            'scale_factor': self.scale_factor,
        }
