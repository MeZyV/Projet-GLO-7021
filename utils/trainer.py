import torch
from tqdm.auto import tqdm
from utils.points import cords_to_map

def train_synthetic_magic(model, optimizer, loss_fn, train_dataloader, valid_dataloader, writer, save_path, filename, epochs, saver_every=None, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        for iter, batch in enumerate(pbar):
            
            im = batch['image'].to(device).type(torch.float)
            label = batch['landmarks'].to(device)
            
            #go through model
            chi_points, _ = model(im)
            
            #get map label
            label = label.type(torch.double)
            size = im.size()
            map = cords_to_map(label, size)
            map[map<0.005] = 0
            
            #loss
            optimizer.zero_grad()
            loss = loss_fn(map, chi_points, device=device)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss, epoch * len(train_dataloader) + iter)
            
            pbar.set_description(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_dataloader)
            for iter, batch in enumerate(pbar):
                im = batch['image'].to(device).type(torch.float)
                label = batch['landmarks'].to(device)
                
                #go through model
                chi_points, _ = model(im)
                
                #get map label
                label = label.type(torch.double)
                size = im.size()
                map = cords_to_map(label, size)
                map[map<0.005] = 0
                
                #loss
                loss = loss_fn(map, chi_points, device=device)
                writer.add_scalar("Loss/valid", loss, epoch * len(valid_dataloader) + iter)
                
                pbar.set_description(f"Epoch: {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")
            
        if saver_every and epoch % saver_every == 0:
            torch.save(model.state_dict(), save_path + filename)
    
    # Save weights
    torch.save(model.state_dict(), save_path + filename)