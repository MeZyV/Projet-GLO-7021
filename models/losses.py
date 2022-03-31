import torch
import torch.nn as nn
import kornia as K
import torch.nn.functional as F


def detector_loss(true_map, chi, v_mask=None):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  n, c, h, w = true_map.size()
  block_size = 8
  true_map = true_map.type(torch.float)
  unfolded_map = torch.nn.functional.unfold(true_map, block_size, stride=block_size)
  unfolded_map = unfolded_map.view(n, c * block_size ** 2, h // block_size, w // block_size)
  unfolded_map = unfolded_map.permute(0,2,3,1)
  shape = torch.cat([torch.tensor(unfolded_map.size())[:3], torch.tensor([1])], dim=0)
  unfolded_map = torch.cat([2*unfolded_map, torch.ones(tuple(shape)).to(DEVICE)], dim=3)
  noise = torch.rand(unfolded_map.size())*0.1
  noise = noise.to(DEVICE)
  label = torch.argmax(unfolded_map+noise,dim=3)
  #define valid mask
  if not v_mask is None:
    valid_mask = v_mask.type(torch.float32).to(DEVICE)
  else:
    valid_mask = torch.ones_like(true_map, dtype=torch.float32).to(DEVICE)
  # adjust valid_mask
  valid_mask = F.unfold(valid_mask, block_size, stride=block_size)
  valid_mask = valid_mask.view(n, c * block_size ** 2, h // block_size, w // block_size)
  valid_mask = valid_mask.permute(0,2,3,1)
  valid_mask = torch.prod(valid_mask, dim=3)
  label[valid_mask==0] = 65
  #get loss
  m = torch.nn.LogSoftmax(dim=1)
  loss = torch.nn.NLLLoss(ignore_index=65)
  output = loss(m(chi), label)
  return output

import kornia as K
import torch.nn.functional as F



def detector_loss(true_map, chi, v_mask=None):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  n, c, h, w = true_map.size()
  block_size = 8
  true_map = true_map.type(torch.float)
  unfolded_map = torch.nn.functional.unfold(true_map, block_size, stride=block_size)
  unfolded_map = unfolded_map.view(n, c * block_size ** 2, h // block_size, w // block_size)
  unfolded_map = unfolded_map.permute(0,2,3,1)
  shape = torch.cat([torch.tensor(unfolded_map.size())[:3], torch.tensor([1])], dim=0)
  unfolded_map = torch.cat([2*unfolded_map, torch.ones(tuple(shape)).to(DEVICE)], dim=3)
  noise = torch.rand(unfolded_map.size())*0.1
  noise = noise.to(DEVICE)
  label = torch.argmax(unfolded_map+noise,dim=3)
  #define valid mask
  if not v_mask is None:
    valid_mask = v_mask.type(torch.float32).to(DEVICE)
  else:
    valid_mask = torch.ones_like(true_map, dtype=torch.float32).to(DEVICE)
  # adjust valid_mask
  valid_mask = F.unfold(valid_mask, block_size, stride=block_size)
  valid_mask = valid_mask.view(n, c * block_size ** 2, h // block_size, w // block_size)
  valid_mask = valid_mask.permute(0,2,3,1)
  valid_mask = torch.prod(valid_mask, dim=3)
  label[valid_mask==0] = 65
  #get loss
  m = torch.nn.LogSoftmax(dim=1)
  loss = torch.nn.NLLLoss(ignore_index=65)
  output = loss(m(chi), label)
  return output