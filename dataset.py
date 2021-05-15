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



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, particles, counts, transform=None, seed=1234):
        self.root_dir = root_dir
        self.transform = transform
        
        self.particles = particles
        self.counts = counts

        self.labels = []
        self.data = []

        for particle in self.particles:
            for count in self.counts:
                data_dir = f'{self.root_dir}/SPI_{particle}_1k_{count}_thumbnail.h5'

                # Load images as h5 files
                f = h5py.File(data_dir, 'r')
                dset_name = list(f.keys())[0]
                data = f[dset_name]
                data = [Image.fromarray(data[i]) for i in range(LENGTH)]
                data = [self.transform(data[i]) for i in range(LENGTH)]
                # label = np.array([1 if i==self.counts.index(count) else 0 for i in range(len(self.counts))])
                # label = np.array([label for _ in range(LENGTH)])
                label = [self.counts.index(count)] * LENGTH
                self.data.extend(data)
                self.labels.extend(label)
        
        # Shuffle the data
        random.seed(seed)
        perm = list(range(len(self.data)))
        random.shuffle(perm)
        self.data = [self.data[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.data)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        X = self.data[index]
        y = self.labels[index]
        return X, y


      
      
def get_dataloaders(args, train_val_particles, test_particles, test_diff_particle=False):
    transform = transforms.Compose([transforms.CenterCrop(128),
                                    transforms.ToTensor()])
    
    if not test_diff_particle:
        assert train_val_particles == test_particles
        dataset = CustomDataset(root_dir=args.root_dir,
                                particles=train_val_particles,
                                counts=COUNTS,
                                transform=transform)
        train_idx = list(range(0, 7000))
        valid_idx = list(range(7000, 8000))
        test_idx = list(range(8000, 9000))
        train_dataset = Subset(dataset, train_idx) 
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)
    else:
        # Create train/valid/test datasets
        train_val_dataset = CustomDataset(root_dir=args.root_dir, 
                                          particles=train_val_particles,
                                          counts=COUNTS,
                                          transform=transform)
        train_idx = list(range(0, 7000))
        valid_idx = list(range(7000, 8000))
        train_dataset = Subset(train_val_dataset, train_idx) 
        valid_dataset = Subset(train_val_dataset, valid_idx)
        test_dataset = CustomDataset(root_dir=args.root_dir, 
                                    particles=test_particles,
                                    counts=COUNTS,
                                    transform=transform)
        
        assert train_dataset.__getitem__(0)[0].shape == torch.Size([1, 128, 128])
        assert valid_dataset.__getitem__(0)[0].shape == torch.Size([1, 128, 128])
        assert test_dataset.__getitem__(0)[0].shape == torch.Size([1, 128, 128])

    # Create train/valid/test dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle, 
                                  num_workers=args.num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle, 
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=args.shuffle, 
                                 num_workers=args.num_workers)
    return train_dataloader, valid_dataloader, test_dataloader
