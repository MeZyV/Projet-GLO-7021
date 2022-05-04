import torch
import numpy as np
from tqdm.auto import tqdm
from utils.points import cords_to_map
from kornia.geometry.transform import warp_perspective


def extract_map(label, size, th=0.005, device='cpu'):
    # get map label
    label = label.type(torch.double)
    map_ = cords_to_map(label, size)
    map_[map_ < th] = 0
    return map_.type(torch.float).to(device)


def train_synthetic_magic(model, optimizer, loss_fn, train_dataloader, valid_dataloader, writer, save_path, filename, epochs, saver_every=None, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        for iter, batch in enumerate(pbar):
            im = batch['image'].to(device).type(torch.float)
            
            #go through model
            chi_points, _ = model(im)
            
            #get map label
            map0 = extract_map(batch['landmarks'], im.size(), device=device)
            
            #loss
            optimizer.zero_grad()
            loss = loss_fn(map0, chi_points, device=device)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch * len(train_dataloader) + iter)
            
            pbar.set_description(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_dataloader)
            for iter, batch in enumerate(pbar):
                im = batch['image'].to(device).type(torch.float)
                
                #go through model
                chi_points, _ = model(im)
                
                #get map label
                map0 = extract_map(batch['landmarks'], im.size(), device=device)
                
                #loss
                loss = loss_fn(map0, chi_points, device=device)
                writer.add_scalar("Loss/valid", loss, epoch * len(valid_dataloader) + iter)
                
                pbar.set_description(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")
            
        if saver_every and epoch % saver_every == 0:
            torch.save(model.state_dict(), save_path + filename + f'_{epoch}.pt')
    
    # Save weights
    torch.save(model.state_dict(), save_path + filename + '.pt')


def train_homography(model, optimizer, loss_fns, train_dataloader, valid_dataloader, writer, save_path, filename,
                          epochs, saver_every=None, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        for iter, batch in enumerate(pbar):
            im = batch['image'].to(device).type(torch.float)
            im_twin = batch['twin_im'].to(device).type(torch.float)
            size = im.size()

            # go through model
            semi0, desc0 = model(im)
            semi1, desc1 = model(im_twin)

            # get map label
            map0 = extract_map(batch['landmarks'], size, device=device)
            map1 = warp_perspective(map0, batch['homography'].to(device), size[-2:], flags='nearest')

            # loss
            optimizer.zero_grad()
            loss = loss_fns[0](map0, semi0, device=device)
            loss += loss_fns[0](map1, semi1, device=device)
            loss += loss_fns[1](desc0, desc1, batch['homography'], device=device)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch * len(train_dataloader) + iter)

            pbar.set_description(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_dataloader)
            for iter, batch in enumerate(pbar):
                im = batch['image'].to(device).type(torch.float)
                im_twin = batch['twin_im'].to(device).type(torch.float)
                size = im.size()

                # go through model
                semi0, desc0 = model(im)
                semi1, desc1 = model(im_twin)

                # get map label
                map0 = extract_map(batch['landmarks'], size, device=device)
                map1 = warp_perspective(map0, batch['homography'].to(device), size[-2:], flags='nearest')

                # loss
                loss = loss_fns[0](map0, semi0, device=device)
                loss += loss_fns[0](map1, semi1, device=device)
                loss += loss_fns[1](desc0, desc1, batch['homography'], device=device)
                writer.add_scalar("Loss/valid", loss, epoch * len(valid_dataloader) + iter)

                pbar.set_description(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

        if saver_every and epoch % saver_every == 0:
            torch.save(model.state_dict(), save_path + filename + f'_{epoch}.pt')

    # Save weights
    torch.save(model.state_dict(), save_path + filename + '.pt')