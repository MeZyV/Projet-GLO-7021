import cv2
import time
import torch
import pandas as pd
from torchvision.transforms import ToTensor, Compose, ToPILImage, Grayscale
from torch.utils.data import Subset
from tqdm.auto import tqdm

from datasets.synthetic import SyntheticShapes_dataset
from models.superpoint import SuperPointNet
from utils.functions import topk_2d

########################################################################################################################
DEVICE = 'cpu'
N_IMAGE = 1000
NUM_KP = 60

########################################################################################################################
model = SuperPointNet(superpoint_bool=False)
model.load_state_dict(torch.load('models/weights/base_detector.pt'))
model.eval()
model.to(DEVICE)

orb = cv2.ORB_create(nfeatures=NUM_KP)
sift = cv2.SIFT_create(nfeatures=NUM_KP)

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

mean_model = 0
mean_model_gpu = 0
mean_orb = 0
mean_sift = 0

for i, batch in enumerate(tqdm(dataset)):
    with torch.no_grad():
        model.cpu()
        im = transform(batch['image'])
        im = im.to(DEVICE).type(torch.float)
        im = im.unsqueeze(0)
        label = batch['landmarks'].to(DEVICE)
        label = label.squeeze(0)
        label = label[:, :2]
        
        #go through model
        start = time.time()
        kp0, _ = model(im, dense=False)
        time_model += time.time() - start
        kp0 = kp0.squeeze(0).squeeze(0)
        val0, kp0 = topk_2d(kp0, NUM_KP)
        
        #go through model on gpu
        model.cuda()
        start = time.time()
        kp1, _ = model(im.cuda(), dense=False)
        time_model_gpu += time.time() - start
        kp1 = kp1.squeeze(0).squeeze(0)
        val0, kp1 = topk_2d(kp1, NUM_KP)

        #go through orb
        start = time.time()
        kp2, _ = orb.detectAndCompute(batch['image'], None)
        time_orb += time.time() - start
        kp2 = torch.tensor([kp.pt for kp in kp2]).type(torch.int)
        if kp2.shape[0] == 0:
            kp2 = torch.zeros((1, 2))
        kp2 = kp2[:NUM_KP, :]
        pad = torch.zeros(NUM_KP-kp2.shape[0], 2)
        kp2 = torch.cat((kp2, pad), dim=0)

        #go through sift
        start = time.time()
        kp3, _ = sift.detectAndCompute(batch['image'], None)
        time_sift += time.time() - start
        kp3 = torch.tensor([kp.pt for kp in kp3]).type(torch.int)
        if kp3.shape[0] == 0:
            kp3 = torch.zeros((1, 2))
        kp3 = kp3[:NUM_KP, :]
        pad = torch.zeros(NUM_KP-kp3.shape[0], 2)
        kp3 = torch.cat((kp3, pad), dim=0)

        #Calculate distance between points  
        mean_model += torch.mean(torch.norm(kp0 - label, dim=1)).item()
        mean_model_gpu += torch.mean(torch.norm(kp1 - label, dim=1)).item()
        mean_orb += torch.mean(torch.norm(kp2 - label, dim=1)).item()
        mean_sift += torch.mean(torch.norm(kp3 - label, dim=1)).item()

res = pd.DataFrame(data={
    'model': [mean_model/len(dataset), time_model/len(dataset)],
    'model_gpu': [mean_model_gpu/len(dataset), time_model_gpu/len(dataset)],
    'orb': [mean_orb/len(dataset), time_orb/len(dataset)],
    'sift': [mean_sift/len(dataset), time_sift/len(dataset)]
}, index=['mean_norm', 'time/img']).T

print(res.to_latex())
