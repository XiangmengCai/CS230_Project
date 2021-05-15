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


def train(args, model, optimizer, loss_fn):
    """""
    Train the network on the training data
    """
    from tqdm import tqdm

    EPOCH = args.epoches

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(args, 
                                                                          PARTICLES, 
                                                                          PARTICLES,
                                                                          test_diff_particle=False)

    step = 0

    train_loss_values = []
    # valid_loss_values = []
    for epoch in range(EPOCH):
        epoch_train_loss = 0.0
        with tqdm(total=len(train_dataloader)) as t: 
            for i, (inputs, labels) in enumerate(train_dataloader):
                step += 1
                model.train()

                inputs = inputs.to(device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.squeeze(0).to(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cost = loss.item()

                epoch_train_loss = cost
                
                t.set_postfix(train_loss='{:05.3f}'.format(cost))
                t.update()
        
        if epoch % args.evaluate_every == 0:    
            accuracy, valid_loss = evaluate(model, criterion, valid_dataloader)
            # epoch_valid_loss = valid_loss
            print(f'Step {step}: valid loss={valid_loss}, valid accuracy={accuracy}')

        # train_loss_values.append(epoch_train_loss)
        valid_loss_values.append(epoch_valid_loss)

    torch.save(model.state_dict(), args.save_path) 

    """
    Plot the epoch loss
    """
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    # plt.plot(valid_loss_values,label="val", color='r')
    plt.plot(train_loss_values,label="train", color='b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    """
    Report results on the train and test data (using the evaluation metric)
    """
    
    train_accuracy, _ , _, _ = calculate_accuracy(train_dataloader)
    test_accuracy, images, preds, labels = calculate_accuracy(test_dataloader)

    print('Train accuracy: %f' % train_accuracy)
    print('Test accuracy: %f' % test_accuracy)

    images = np.concatenate(images, axis=0)
    preds = np.concatenate(preds, axis=0)
    labels = np.squeeze(np.concatenate(labels, axis=0))

    return images, preds, labels
  
if __name__ == '__main__':
    args = parser.parse_args()
    
    model = load_model(args)
    
    # Define the cost function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer, learning rate 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    images, preds, labels = train(args, model, optimizer, criterion)
