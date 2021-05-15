import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
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


def evaluate(model, loss_fn, dataloader):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    """
    model.eval()

    accuracies = []
    loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        preds = model(images)
        loss += loss_fn(preds, labels.to(device))
        
        preds = torch.argmax(preds, dim=-1)
        assert preds.shape == labels.shape
        correct_pred = [1 if preds[i] == labels[i] else 0 for i in range(preds.shape[0])]
        acc = sum(correct_pred) / len(preds)

        accuracies.append(acc)
        
    loss = loss / len(dataloader)
    accuracy = np.mean(accuracies)

    return accuracy, loss

def calculate_accuracy(loader):
    """
    Define evaluation metric
    We will use accuracy as an evaluation metric
    """
    total = 0
    correct = 0
  
    all_images = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze().to(device)).sum().item()
            
            all_images.append(images)
            all_preds.append(predicted.cpu().data.numpy())
            all_labels.append(labels)

    return 100 * correct / total, all_images, all_preds, all_labels
