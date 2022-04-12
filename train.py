import torch
import kornia as K
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter, Grayscale, GaussianBlur, ToPILImage
from torch.utils.tensorboard import SummaryWriter

from datasets.synthetic import SyntheticShapes_dataset
from utils.plot import plot_imgs
from utils.points import cords_to_map
from utils.train import train_synthetic_magic
from models.superpoint import SuperPointNet
from models.losses import DectectorLoss

train_on_gpu = torch.cuda.is_available()
print(torch.__version__)

if not train_on_gpu:
    DEVICE = 'cpu'
    print('CUDA is not available.  Training on CPU ...')
else:
    DEVICE = 'cuda'
    print('CUDA is available!  Training on GPU ...')

transform = Compose([
    ToPILImage(),
    ColorJitter(brightness=(0.5,1.4)),
    Grayscale(num_output_channels=1),
    GaussianBlur(9, sigma=(1, 1.8)),
    ToTensor()
])

landmark_transform = Compose([
    ToTensor()
])

dataset = SyntheticShapes_dataset(
    csv_file='./data/synthetic_shapes/syn_shape_labels.csv',
    root_dir='./data/synthetic_shapes/images/', 
    transform=transform, 
    landmark_transform=landmark_transform,
    landmark_bool=True
)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

print(len(dataset))
print(len(dataloader))

writer = SummaryWriter(f'./logs/magic_train/{datetime.now().strftime("%m%d-%H%M")}')

model = SuperPointNet(superpoint_bool=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = DectectorLoss()

train_synthetic_magic(
    model=model, 
    optimizer=optimizer,
    loss_fn=loss_fn, 
    dataloader=dataloader,
    writer=writer, 
    save_path='.', 
    filename='test.pt',
    device=DEVICE
)
