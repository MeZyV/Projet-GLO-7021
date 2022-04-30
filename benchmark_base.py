import cv2
import time
import torch
import pandas as pd
from torchvision.transforms import ToTensor, Compose, ToPILImage, Grayscale
from torch.utils.data import Subset
from tqdm.auto import tqdm

from datasets.synthetic import SyntheticShapes_dataset
from models.superpoint import SuperPointNet
from models.losses import DectectorLoss
from utils.points import cords_to_map, map_to_cords

########################################################################################################################
DEVICE = 'cpu'
N_IMAGE = 100

########################################################################################################################
loss_fn = DectectorLoss()

########################################################################################################################
model = SuperPointNet(superpoint_bool=False)
model.load_state_dict(torch.load('models/weights/base_detector.pt'))
model.eval()
model.to(DEVICE)

orb = cv2.ORB_create(nfeatures=60)
sift = cv2.SIFT_create(nfeatures=60)

########################################################################################################################
transform = Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),
    ToTensor()
])

dataset = SyntheticShapes_dataset(
    csv_file='./data/synthetic_shapes/syn_shape_labels.csv',
    root_dir='./data/synthetic_shapes/images/',
    transform=None,
    landmark_transform=ToTensor(),
    landmark_bool=True
)
dataset = Subset(dataset, range(len(dataset) - N_IMAGE, len(dataset)))

print(f'Testing on {len(dataset)} images')

########################################################################################################################
time_model = 0
time_model_gpu = 0
time_orb = 0
time_sift = 0

for i, batch in enumerate(tqdm(dataset)):
    with torch.no_grad():
        model.cpu()
        im = transform(batch['image'])
        im = im.to(DEVICE).type(torch.float)
        im = im.unsqueeze(0)
        label = batch['landmarks'].to(DEVICE)
        
        #go through model
        start = time.time()
        kp0, _ = model(im)
        time_model += time.time() - start

        #go through orb
        start = time.time()
        kp1, _ = orb.detectAndCompute(batch['image'], None)
        time_orb += time.time() - start

        #go through sift
        start = time.time()
        kp2, _ = sift.detectAndCompute(batch['image'], None)
        time_sift += time.time() - start

        #go through model on gpu
        model.cuda()
        start = time.time()
        kp3, _ = model(im.cuda())
        time_model_gpu += time.time() - start
        
        #get map label
        # label = label.type(torch.double)
        # size = im.size()
        # map = cords_to_map(label, size)
        # map[map<0.005] = 0
        
        #loss
        # loss_model = loss_fn(map, kp0, device=DEVICE)
        
        # responses = torch.Tensor([k.response for k in kp2])
        # kp2 = torch.Tensor([k.pt for k in kp2]).type(torch.int)

res = pd.DataFrame(data={
    'model': [time_model, time_model/len(dataset)],
    'model_gpu': [time_model_gpu, time_model_gpu/len(dataset)],
    'orb': [time_orb, time_orb/len(dataset)],
    'sift': [time_sift, time_sift/len(dataset)]
}, index=['time', 'time/img']).T

print(res)