# Name: Ting Li; Duo Xu. 
# Date: March 27.
# Travel day used: 2 days totally, 1 day for Ting and 1 day for Duo. 
# This file has the functions for training and evaluating the experiment network.

import os
import torch
import torch.optim as optim
import torch.nn.functional as F

# train the network and count the average loss
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# count the accuracy of the model
def evaluate_accuracy(model, device, test_loader):
    model.eval()  
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

# save the trained model
def save_model(model, save_dir, model_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, model_name))

