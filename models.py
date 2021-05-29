import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, DataLoader

import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from argparse import Namespace

import random
import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
  
  
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
                
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        
    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1) # x = [batch size, height * width]
        h_1 = F.relu(self.input_fc(x)) # h_1 = [batch size, 250]
        h_2 = F.relu(self.hidden_fc(h_1)) # h_2 = [batch size, 100]
        y_pred = self.output_fc(h_2) # y_pred = [batch size, output dim]
        return y_pred
       
def create_resnet18():
    resnet18 = models.resnet18(pretrained=False)
    resnet18.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    retrun resnet18

def create_vgg16():
    vgg16 = models.vgg16(pretrained=False)
    vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return vgg16
    
def create_alexnet():
    alexnet = models.alexnet(pretrained=False)
    alexnet.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return alexnet

  
def get_model(args):
    if args.model == 'mlp':
        input_dim = 128 * 128
        output_dim = 3
        model = MLP(input_dim, output_dim).to(device)
    elif args.model == 'vgg16':
        vgg16 = create_vgg16()
        model = vgg16.to(device)
    elif args.model == 'resnet18':
        resnet18 = create_resnet18()
        model = resnet18.to(device)
    elif args.model == 'alexnet':
        alexnet = create_alexnet()
        model = alexnet.to(device)
    return model
