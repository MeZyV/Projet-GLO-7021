import torch
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter

from transforms import ToGray
from datasets.coco import COCO
from utils.trainer import train_homography
from models.superpoint import SuperPointNet
from models.superattentionpoint import SuperAttentionPointNet
from models.losses import DectectorLoss, DescriptorLoss

print(f'PyTorch version : {torch.__version__}')

if torch.cuda.is_available():
    DEVICE = 'cuda'
    print('Training on GPU.')
else:
    DEVICE = 'cpu'
    print('Training on CPU.')
DEVICE = 'cpu'
print('Training on CPU.')

transform = Compose([
    Resize((240, 320)),
    ToTensor(),
    ToGray(False)
])

landmark_transform = Compose([
    ToTensor()
])

dataset = COCO(
    csv_file='./data/coco/coco_val_attention_labels.csv',
    root_dir='./data/coco/val2017/',
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

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

print(f'Training on {len(train_dataset)} images')
print(f'Validation on {len(valid_dataset)} images')

writer = SummaryWriter(f'./logs/ms_coco_train/{datetime.now().strftime("%m%d-%H%M")}')

MODEL_ = "Attention"  # "SuperPoint"
if MODEL_ == "Attention":
    model = SuperAttentionPointNet(embed_dim=512, hidden_dim=768, num_heads=(4, 8, 4, 2),
                                   patch_size=8, img_size=(240, 320), desc=True)
else:
    model = SuperPointNet(superpoint_bool=False)
print(MODEL_)
print(model)

weights_superpoint = './models/weights/base_detector_Attention_6.pt'
model.load_state_dict(torch.load(weights_superpoint))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fns = (DectectorLoss(), DescriptorLoss())

train_homography(
    model=model,
    optimizer=optimizer,
    loss_fns=loss_fns,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    writer=writer,
    save_path='./models/weights/',
    filename='base_detector_' + MODEL_,
    epochs=10,
    saver_every=1,
    device=DEVICE
)
