import time
import torch 
from utils.points import cords_to_map

def train_synthetic_magic(model, optimizer, loss_fn, dataloader, writer, save_path, filename, device='cpu'):
    t_0 = time.time()
    model.to(device)
    model.train()
    for e in range(20):
        for iter, batch in enumerate(dataloader):
            
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
            writer.add_scalar("Loss/train", loss, e*len(dataloader)+iter)
            
            if iter%10==0:
                print('iteration {}/{} is running'.format(e*len(dataloader)+iter,20*len(dataloader)))
                print('loss is:',loss.item())
            if iter % 50 ==0:
                t_c = time.time()
                minute = (t_c-t_0)/60
                print('saving weights from iteration {} with loss {}, {} minutes pased'.format(e*len(dataloader)+iter,loss.item(),int(minute)))
                torch.save(model.state_dict(), save_path+filename)
    
    # Save weights
    torch.save(model.state_dict(), save_path+filename)
    t_f = time.time()
    hours = (t_f-t_0)/3600
    print('finished in {} hours'.format(hours))