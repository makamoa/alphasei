from utils import *
import torch 
from torch import nn 
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Any, Tuple
import tqdm

device = "cpu"
if torch.cuda.is_available():
	device = torch.device("cuda")   
	print("Running on: Cuda")
else:
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
        print ("Running on mps")
        

def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                metrics: Any,
                epoch: int,
                writer: torch.utils.tensorboard.SummaryWriter) -> float:
    model.train()
    metrics.reset()
    total_loss = 0
    
    # print ("About to load the data:")
    for i, (data, target) in tqdm.tqdm(enumerate(dataloader)):
        # print("Got the data")
        data, target = data.to(device), target.to(device)
        # shape the data as (batchsize, channel, rest)
        
        # print (data.shape)
        
        optimizer.zero_grad()
        output = model(data)

        loss_output = criterion(output, target)
        loss_output.backward()
        total_loss += loss_output.item()
        optimizer.step()
        
        # print(output.shape)
        
        sf = nn.Softmax(1)
        preds = sf(output.detach())
        # print(preds.shape)
        preds = torch.argmax(preds, dim = 1)
        # print(preds.shape)
        
        metrics.update(preds.detach(), target)
        
        if i % 5 == 0:
            writer.add_scalar('Loss/train_step', loss_output.item(), epoch*len(dataloader) + i)
    
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    metrics.to_tensorboard(writer, epoch, 'train')
    
    return avg_loss

def validate(model: nn.Module,
             dataloader: torch.utils.data.DataLoader,
             criterion: nn.Module, 
             metrics: Any,
             epoch: int,
             writer: torch.utils.tensorboard.SummaryWriter) -> float:
    model.eval()
    metrics.reset()
    total_loss = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss_output = criterion(output, target)
            total_loss += loss_output.item()
            
            sf = nn.Softmax(1)
            preds = sf(output.detach())
        # print(preds.shape)
            preds = torch.argmax(preds, dim = 1)
            
            metrics.update(preds, target)
        
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
        metrics.to_tensorboard(writer, epoch, 'val')
        
        return avg_loss
 
def train(model: nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          metrics: Any,
          num_epochs: int,
          writer: torch.utils.tensorboard.SummaryWriter,
          save_dir: str) -> Tuple[list, list]:
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, metrics, epoch, writer)
        val_loss = validate(model, val_loader, criterion, metrics, epoch, writer)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            print(f'Validation loss decreased from {best_val_loss} to {val_loss}. Epoch {epoch}, Saving model...')
            best_val_loss = val_loss
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            
        elif (epoch + 1) % 10 == 0:
            print (f'Saving checkpoint at epoch {epoch}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
                    
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Time: {time.time() - start_time}')
    
    return train_losses, val_losses


if __name__ == "__main__":
    # python3 train.py --model mode --modelConfig modelConfigl --dsType dataset --datasetConfig datasetConfig --trainConfig trainConfig
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelType', type=str, help='model name')
    parser.add_argument('--modelConfig', type=str, help='model config json file')
    parser.add_argument('--datasetType', type=str, help='dataset type: segy, npy')
    parser.add_argument('--datasetConfig', type=str, help='dataset config json file')
    parser.add_argument('--trainConfig', type=str, help='train config json file')
    args = parser.parse_args()
        
    model = get_model(args.modelType, args.modelConfig)
    model = model.to(device)
    
    criterion, num_epochs, optimizer, scheduler, metrics, (td, tl) = train_config(model, args.trainConfig)
    
    train_loader, val_loader = get_loaders(args.datasetType, args.datasetConfig, td, tl)
    
    # exit()
    save_dir = os.path.join('runs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)
    
    # Train the model
    tl, vl = train(model, train_loader, val_loader, criterion, optimizer, scheduler, metrics, num_epochs, writer, save_dir)
    
    plot_losses(tl, vl, save_dir)
    
    writer.close()