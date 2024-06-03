import os
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

# *Reusing my previous code

## Saving and Loading Models
def text_to_file(JOINED_PATH:str, contents, mode:str='w'):
    with open(JOINED_PATH, mode) as f:
        print(contents, file=f)

def save_model(model:nn.Module, PATH:str='models', FILENAME:str=None, extra_info:str=""):
    if FILENAME == None:
        import time
        FILENAME = f'model_{int(time.time())}.h5'

    import os
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    PATH = os.path.join(PATH, FILENAME)
    torch.save(model.state_dict(), PATH)

    text_to_file(PATH+'_details.txt', str(model)+'\n'+extra_info)
    return PATH

def save(TRAIN_ID, model, training_accuracy, training_losses, validation_accuracy, validation_losses, lrs):
    extra_info = f" Train accuracy: {(100*training_accuracy[-1]):>0.1f}%, Avg loss: {training_losses[-1]:>8f}, lr: {lrs[-1]}"
    if validation_accuracy is not None:
        extra_info += f"\n Test accuracy: {(100*validation_accuracy[-1]):>0.1f}%"
        if validation_losses is not None: 
            extra_info += f", Avg loss: {validation_losses[-1]:>8f}"
    return save_model(model=model, PATH=os.path.join('models',str(TRAIN_ID)), extra_info=extra_info)

def load_model(model:nn.Module, FILE_PATH:str, device='cpu'):
    if FILE_PATH is None:
        return
    model.load_state_dict(torch.load(FILE_PATH, map_location=torch.device(device)))

## Related to Training
def set_optimizers(model:nn.Module, loss_fn=nn.CrossEntropyLoss, optimizer=torch.optim.SGD, lr=1e-1, decay=None, **kwargs):
    # Loss function
    loss_fn = loss_fn()

    # SGD Optimizer
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = decay(optimizer, **kwargs) if decay != None else None
    return loss_fn, optimizer, scheduler

# Train function
def train(dataloader:DataLoader, model:nn.Module, loss_fn, optimizer, scheduler=None, device='cpu', log:bool=True):
    size = len(dataloader.dataset)
    
    # Turn on training mode
    model.to(device)
    model.train()
    train_loss, correct = 0, 0
    dt = tqdm(dataloader) if log else dataloader
    for X, y in dt:
        X, y = X.to(device), y.to(device).float()

        # Compute prediction error
        
        # print(X.shape, y.shape)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss
        train_loss += loss.item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    
    train_loss /= len(dataloader)
    correct /= size
    lr = optimizer.param_groups[0]['lr']
    if scheduler != None:
        scheduler.step() if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step(loss)
    
    if log:
        print(f" Train accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}, lr: {lr}")
    return train_loss, correct, lr

# Test function
def test(dataloader:DataLoader, model:nn.Module, loss_fn, device='cpu', log:bool=True):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    # Turn on evalution mode
    model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    
    # Turn off gradient descent
    with torch.no_grad():
        dt = tqdm(dataloader) if log else dataloader
        for X, y in dt:
            X, y = X.to(device), y.to(device).float()
            pred = model(X)
            
            # record loss
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    
    if log:
        print(f" Test accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct

# Generating Output
def gen_output(dataloader:DataLoader, model:nn.Module, device='cpu'):    
    # Turn on evalution mode
    model.to(device)
    model.eval()

    names = list()
    predictions = list()
    
    # Turn off gradient descent
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y
            pred = model(X)
            p = pred.argmax(1).cpu().numpy()
            p = p.tolist()
            
            names.extend(y)
            predictions.extend(p)
            
    return names, predictions