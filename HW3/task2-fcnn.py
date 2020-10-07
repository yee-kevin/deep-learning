# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
from torch.utils.data.sampler import SubsetRandomSampler

# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms

# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json
from IPython.display import clear_output

import numpy as np
import collections
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)     # On by default, leave it here for clarity

# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)

#training batch size
batch_size=8
#validation batch size
valbatch_size=8

indices = list(range(len(train_set)))
train_indices, val_indices = indices[:50000], indices[50000:]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

test_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)

train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
        train_set, batch_size=valbatch_size, sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=valbatch_size, sampler=None)

# Build the neural network, expand on top of nn.Module
class Network(nn.Module):
  def __init__(self):
    super().__init__()

    # define layers    
    self.fc1 = nn.Linear(in_features=1*28*28, out_features=300)
    self.fc2 = nn.Linear(in_features=300, out_features=100)
    self.out = nn.Linear(in_features=100, out_features=10)

  # define forward function
  def forward(self, t):
    # Use view() to get [batch_size, num_features].
    # -1 calculates the missing value given the other dim.
    t = t.view(batch_size, -1) # torch.Size([1, 784])
    # fc1
    t = self.fc1(t)
    t = torch.nn.functional.relu(t)

    # fc2
    t = self.fc2(t)
    t = torch.nn.functional.relu(t)

    # output
    t = torch.nn.functional.log_softmax(self.out(t), dim=1)
    # don't need softmax here since we'll use cross-entropy as activation.

    return t

def train_epoch(model,  trainloader,  criterion, device, optimizer ):
    
    model.train() 
    
    # Total loss / Total correct
    total_loss = 0.0   
    total_correct = 0
    
    for batch_idx, data in enumerate(trainloader):
        inputs=data[0].to(device)
        labels=data[1].to(device) 
        
        # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls)
        optimizer.zero_grad()
        
        # calculate and add correct predictions vs labels
        outputs = model(inputs) 
        cpuout= outputs.to('cpu')                  
        _, preds = torch.max(cpuout, 1) 
        total_correct += torch.sum(preds == labels.data)
        
        # add loss
        loss = criterion(outputs, labels)     
        total_loss += loss
        
        # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        loss.backward()            
        optimizer.step()         
            
    total_size = trainloader.__len__()*trainloader.batch_size*1.0    
    acc = total_correct.double() / total_size
    avg_loss = total_loss / (trainloader.__len__()*1.0)
    
    return acc.item(), avg_loss.item()

def evaluate(model, dataloader, criterion, device):   
    
    model.eval()

    total_loss = 0.0   
    total_correct = 0
    
    with torch.no_grad():
      for ctr, data in enumerate(dataloader):
#           print ('epoch at 10000', ctr)            
          inputs = data[0].to(device) 
          labels = data[1] 
          
          # calculate and add correct predictions vs labels  
          outputs = model(inputs)  
          cpuout= outputs.to('cpu')            
          _, preds = torch.max(cpuout, 1)
          total_correct += torch.sum(preds == labels.data)  
          
          # add loss
          loss = criterion(outputs, labels)          
          total_loss += loss          
      
      total_size = dataloader.__len__()*dataloader.batch_size*1.0
      acc = total_correct.double() / total_size # this does not work if one uses a datasampler!!!
      avg_loss = total_loss / (dataloader.__len__()*1.0)
    
    return acc.item(), avg_loss.item() 


def train_modelcv(dataloader_cvtrain, dataloader_cvval, dataloader_cvtest ,  model ,  criterion, optimizer, scheduler, num_epochs, device):

  graph_data = collections.defaultdict(list)
  bestweights = 0  
  best_val_loss = 100000000000
  best_epoch =-1

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))    

    model.train(True)    
    train_acc, train_loss = train_epoch(model,  dataloader_cvtrain,  criterion,  device , optimizer )    
    #scheduler.step()         

    model.train(False)
    val_acc, val_loss = evaluate(model, dataloader_cvval, criterion, device)    
    
    # IMPORTANT NOTE: I assumed that the instructions in the homework means that it is
    #                 INCORRECT to use the test dataset as the validation dataset. Hence,
    #                 I correctly use the validation dataset.      
    # Choosing the model that gives the best validation loss    
    if val_loss < best_val_loss: #higher is better or lower is better? lower is better
      bestweights= model.state_dict()
      best_val_loss = val_loss
      best_epoch = epoch    
    
    print('Current val_loss:', val_loss, ', Current best_val_loss:', best_val_loss, ' at epoch ', best_epoch)
    print('-' * 10)
    
    model.load_state_dict(bestweights) 
    
    test_acc, test_loss = evaluate(model, dataloader_cvtest, criterion, device)        
    
    graph_data['train_acc'].append(train_acc)
    graph_data['train_loss'].append(train_loss)
    graph_data['val_acc'].append(val_acc)
    graph_data['val_loss'].append(val_loss)
    graph_data['test_acc'].append(test_acc)
    graph_data['test_loss'].append(test_loss)
    
  return graph_data

def plot_graphs(maxnumepochs, graph_data, learning_rate):
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.title('Acc vs Epoch [learning rate:{}]'.format(learning_rate))
    plt.plot(list(range(maxnumepochs)), graph_data['train_acc'], label='train_acc')
    plt.plot(list(range(maxnumepochs)), graph_data['val_acc'], label='val_acc')
    plt.plot(list(range(maxnumepochs)), graph_data['test_acc'], label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()

    plt.subplot(122)
    plt.title('Loss vs Epoch [learning rate:{}]'.format(learning_rate))
    plt.plot(list(range(maxnumepochs)), graph_data['train_loss'], label='train_loss')
    plt.plot(list(range(maxnumepochs)), graph_data['val_loss'], label='val_loss')
    plt.plot(list(range(maxnumepochs)), graph_data['test_loss'], label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    fname = 'graph_lr_{}.png'.format(learning_rate)
    
    plt.savefig(fname, bbox_inches='tight')

    plt.show()

def run(learning_rate):
    model = Network()  

    # negative log likelihood loss
    criterion = torch.nn.NLLLoss()
    
    # number of epoch
    maxnumepochs = 20
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=0.0, weight_decay=0)

    device=torch.device('cpu')
    graph_data = train_modelcv(dataloader_cvtrain = train_loader,                                                          dataloader_cvval = val_loader,  
                               dataloader_cvtest = test_loader,
                               model = model,
                               criterion = criterion,
                               optimizer = optimizer,
                               scheduler = None,
                               num_epochs = maxnumepochs,
                               device = device)    
    plot_graphs(maxnumepochs, graph_data, learning_rate)  

# learning rate 1 = 0.01
learning_rate=0.01
# Run model
run(learning_rate)  

# learning rate 2 = 0.001
learning_rate=0.001
# Run model
run(learning_rate)