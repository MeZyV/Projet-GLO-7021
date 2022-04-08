import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ColorJitter, Grayscale

from datasets.coco import COCO
from PIL import Image

print(f'Using torch v{torch.__version__}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = Compose([
    Resize((240, 320), Image.BICUBIC),
    ColorJitter(brightness=(0.7, 1.3)),
    Grayscale(num_output_channels=1)
])

dataset = COCO(
    csv_file='/datasets/COCO/labeled_coco.csv',
    root_dir='/data/COCO/val2017', 
    transform=transform, 
    landmark_bool=True
)

dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
