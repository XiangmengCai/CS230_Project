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

  
def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def show_pred_examples():
    images, labels, preds = unison_shuffled_copies(images, labels, preds)
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    flatted_axs = [item for one_ax in axs for item in one_ax]
    for ax, img, label, pred in zip(flatted_axs, images[:25], labels[:25], preds[:25]):
        ax.imshow(np.reshape(img, (128, 128)))
        ax.set_title('l:{},p:{}'.format(label, pred))
        ax.axis('off')
    plt.show()
