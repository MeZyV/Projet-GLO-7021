import csv
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from transforms import Rescale, ToTensor, ToGray
from datasets.synthetic import mySyntheticShapes

train_on_gpu = torch.cuda.is_available()
print(torch.__version__)

if not train_on_gpu:
    DEVICE = 'cpu'
    print('CUDA is not available.  Training on CPU ...')
else:
    DEVICE = 'cuda'
    print('CUDA is available!  Training on GPU ...')

transform = Compose([
    Rescale((240, 320)),
    ToTensor(),
    ToGray()
])

config = {
    'primitives': 'all',
    'truncate': {},
    'validation_size': -1,
    'test_size': -1,
    'on-the-fly': False,
    'cache_in_memory': False,
    'suffix': None,
    'add_augmentation_to_test_set': False,
    'num_parallel_calls': 10,
    'generation': {
        'split_sizes': {'training': 10000, 'validation': 200, 'test': 500},
        'image_size': [960,1280], 
        'random_seed': 0,
        'params': {
            'generate_background': {
                'min_kernel_size': 150, 'max_kernel_size': 500,
                'min_rad_ratio': 0.02, 'max_rad_ratio': 0.031}, 
            'draw_stripes': {'transform_params': (0.1, 0.1)},
            'draw_multiple_polygons': {'kernel_boundaries': (50, 100)}
        },
    },
    'preprocessing': {
        'resize': [240, 320],
        'blur_size': 11,
    },
    'augmentation': {
        'photometric': {
            'enable': False,
            'primitives': 'all',
            'params': {},
            'random_order': True,
        },
        'homographic': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
    }
}

num_imgs_returned=10000
dataset = mySyntheticShapes(config, num_imgs_returned, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

#start generating data
filename = './data/synthetic_shapes/syn_shape_labels.csv'
im_path = './data/synthetic_shapes/images/'
with open(filename, 'w') as csvfile:  
    #creating a csv writer object  
    csvwriter = csv.writer(csvfile)
    #write headline
    head = np.array(['x{},y{},conf{}'.format(i, i, i).split(',') for i in range(60)]).reshape((1, -1))
    head = np.concatenate((np.array([['im_name']]), head), axis=1)
    csvwriter.writerows(head)
    
    for iter, (im, label) in enumerate(dataloader):
        print(iter)
        full_cord = torch.flatten(label)
        full_cord = np.asarray(full_cord)
        
        name = 'syn_shape_' + str(iter) + '.jpg'
        full_cord = full_cord.reshape((1, -1))
        new_cords = np.concatenate(([[name]], full_cord), axis=1)
        csvwriter.writerows(new_cords)
        torchvision.utils.save_image(im, im_path + name)