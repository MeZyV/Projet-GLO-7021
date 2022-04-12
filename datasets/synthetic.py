import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from datasets.synthetic_shapes_functions import *


drawing_primitives = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube'
]

class mySyntheticShapes(Dataset):
    def __init__(self, config, num_imgs_returned, transform=None):
        self.config = config 
        self.num_imgs_returned = num_imgs_returned
        self.transform = transform

    def __len__(self):
        return self.num_imgs_returned

    def __getitem__(self, index):
        primitives = parse_primitives(self.config['primitives'], drawing_primitives)
        primitive = np.random.choice(primitives)

        image = generate_background(self.config['generation']['image_size'], **self.config['generation']['params']['generate_background'])
        image = add_gausian_noise(image)

        primitive_func_str = primitive+'(image, **self.config["generation"]["params"].get(primitive, {}))'
        points = np.array(eval(primitive_func_str))
        points = pad_points(points, desired_num_points=60)

        sample = (image,points)

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class SyntheticShapes_dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, landmark_bool=False, landmark_transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_transform = landmark_transform
        self.landmark_bool = landmark_bool

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if self.landmark_bool:
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 3)
        else:
            landmarks = 0

        if self.transform:
            image = self.transform(image)
        if self.landmark_transform:
            landmarks = self.landmark_transform(landmarks)
        
        return {
            'image': image,
            'landmarks': landmarks
        }

