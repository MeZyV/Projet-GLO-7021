import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter, Grayscale, GaussianBlur, ToPILImage
from torch.utils.tensorboard import SummaryWriter

from datasets.synthetic import SyntheticShapes_dataset
from utils.trainer import train_synthetic_magic
from models.superpoint import SuperPointNet
from models.superattentionpoint import SuperAttentionPointNet
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

train_split = len(dataset)
valid_split = int(train_split * 0.1)
train_split = train_split - valid_split
train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset, (train_split, valid_split)
)

train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=12, shuffle=False)

print(f'Training on {len(train_dataset)} images')
print(f'Validation on {len(valid_dataset)} images')

writer = SummaryWriter(f'./logs/magic_train/{datetime.now().strftime("%m%d-%H%M")}')

MODEL_ = "Attention"  # "SuperPoint"
if MODEL_ == "Attention":
    model = SuperAttentionPointNet(embed_dim=512, hidden_dim=768, num_heads=(4, 8, 4, 2),
                                   patch_size=8, img_size=(240, 320))
else:
    model = SuperPointNet(superpoint_bool=False)
print(MODEL_)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = DectectorLoss()

train_synthetic_magic(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    writer=writer,
    save_path='./models/weights/',
    filename='base_detector_' + MODEL_,
    epochs=10,
    saver_every=1,
    device=DEVICE
)
