import torch
from torchvision.transforms import Compose, Resize, ColorJitter
from datasets.coco import COCO

print(f'Using torch v{torch.__version__}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
