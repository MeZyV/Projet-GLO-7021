import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter, Grayscale, GaussianBlur, ToPILImage
from torch.utils.tensorboard import SummaryWriter

from datasets.synthetic import SyntheticShapes_dataset
from utils.trainer import train_synthetic_magic
from models.superpoint import SuperPointNet
from models.losses import DectectorLoss

print(f'PyTorch version : {torch.__version__}')

if torch.cuda.is_available():
    DEVICE = 'cuda'
    print('Training on GPU.')
else:
    DEVICE = 'cpu'
    print('Training on CPU.')

transform = Compose([
    ToPILImage(),
    ColorJitter(brightness=(0.5, 1.4)),
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

print(f'Training on {len(dataset)} images')

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
    save_path='./models/weights',
    filename='base_detector.pt',
    epochs=100,
    saver_every=None,
    device=DEVICE
)
