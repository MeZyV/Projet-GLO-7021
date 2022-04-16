import csv
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import kornia as K

from transforms import ToGray
from datasets.coco import COCO_unsupervised
from models.superpoint import SuperPointNet
from utils.points import map_to_cords, nms

print(f'PyTorch version : {torch.__version__}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {DEVICE} ...')

BATCH_SIZE = 1
N_POINTS = 300

transform = Compose([
    Resize((240, 320)),
    ToTensor(),
    ToGray(False)
])

config = {
    'n_homographies': 50,
    'augmentation': {
        'homographic': {
            'degrees': (-15, 15),
            'translate': (0.3, 0.3),
            'scale': (1.1, 1.5),
            'shear': (-20, 20)
        },
    }
}

filename = './data/coco/coco_val_labels.csv'
im_path = './data/coco/val2017/'
weights_superpoint = './models/weights/superpoint.pth'

dataset = COCO_unsupervised(config, im_path, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
names = dataset.names

model = SuperPointNet(superpoint_bool=False).to(DEVICE)
model.load_state_dict(torch.load(weights_superpoint))
model.eval()

t_0 = time.time()

with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # write headline
    head = np.array(['x{},y{},conf{}'.format(i, i, i).split(',') for i in range(N_POINTS)]).reshape((1, -1))
    head = np.concatenate((np.array([['im_name']]), head), axis=1)
    csvwriter.writerows(head)
    for iter, dict_ in enumerate(dataloader):
        dict_ = {k: torch.flatten(v, 0, 1).to(DEVICE).type(torch.float) for k, v in dict_.items()}
        im, H, H_inv = dict_.values()
        with torch.no_grad():
            # put warps through net
            heatmap, desc = model(im, dense=False)
            #heatmap = heatmap.cpu()

            # invert warping into patches at original location
            n, c, h, w = heatmap.size()
            unwarped_heatmap = K.warp_perspective(heatmap, H_inv, dsize=(h, w))

            # average of heatmaps for each im of a batch
            unwarped_heatmap = torch.stack(unwarped_heatmap.tensor_split(BATCH_SIZE, dim=0), dim=0)
            unwarped_heatmap = torch.mean(unwarped_heatmap, dim=1)

            unwarped_heatmap[:, :, :4, :4] = 0
            unwarped_heatmap[:, :, -4:, :4] = 0
            unwarped_heatmap[:, :, -4:, -4:] = 0
            unwarped_heatmap[:, :, :4, -4:] = 0

            # get points
            points = nms(unwarped_heatmap, 5, topk=N_POINTS)

            # set coordinates for saving
            new_cords = map_to_cords(BATCH_SIZE=BATCH_SIZE, iter=iter, points=points, names=names)

            # writing the data rows
            csvwriter.writerows(new_cords)

            if iter % 100 == 0:
                t_f = time.time()
                minute = round((t_f - t_0) / 60, 3)
                print('iteration {}/{} is runing... {} minutes passed'.format(iter, len(dataloader), minute))


