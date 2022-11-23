import torch
from dataset import GTZAN
import sys, os
import os

import torch.nn as nn

from model import ShallowMusicCNN
dataset = GTZAN('../data/train.pkl')
dataset_test = GTZAN('../data/val.pkl')
print(len(dataset_test))
batcher = dataset.get_batches(batch_size=20, num_batches=None, random_seed=42)

# get 100 test samples
batcher_test = dataset.get_batches(batch_size=20, num_batches=None, random_seed=42)
for spectrograms_test, labels_test in dataset_test.get_batches(batch_size=500, num_batches=None, random_seed=42):
    break
print('size of test set:',spectrograms_test.shape[0])
num_classes = 10
model = ShallowMusicCNN(num_classes)


import numpy as np
from torchmetrics import Accuracy



# utility for getting the current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
accuracy = Accuracy()

# for epoch in range(1000):
for epoch, (spectrograms, labels) in enumerate(batcher):

    output = model(spectrograms)
    if epoch == 0:
        print(output)
    preds = output
    pred_class = preds.argmax(dim=1)
    label_class = labels.argmax(dim=1)
    # print(pred_class)
    loss = F.cross_entropy(preds, labels)

    loss.backward()
    # update weights
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()
    # print("train/epoch_loss", loss)
    # print("train/epoch_accuracy", accuracy(pred_class.int(), label_class.int()))
    # print(model.fc2.weight[0, 0:4],accuracy(pred_class.int(), label_class.int()))
    # values of weight tensors
    wvis = (
        model.fc1.weight[0, 0:2].detach().numpy(),
        model.fc2.weight[0, 0:2].detach().numpy(),
        model.conv1.weight[0, 0, 0, 0:1].detach().numpy(),
        model.conv2.weight[0, 0, 0, 0:1].detach().numpy(),
    )
    if epoch > 100:
        # print('stop showing weights')
        wvis = ''
    print(epoch,
        get_lr(optimizer),
        accuracy(pred_class.int(), label_class.int()).detach().numpy(),
        pred_class.int().detach().numpy(),
        torch.round(loss, decimals=3).detach().numpy(),
        # torch.round(preds.sum(dim=0),decimals=3).detach().numpy(),
            *wvis)
    if epoch % 100 == 0:
        print('test', '='*20)
        with torch.no_grad():
            output = model(spectrograms_test)
            preds = output
            pred_class = preds.argmax(dim=1)
            label_class = labels_test.argmax(dim=1)
            print(epoch,
                get_lr(optimizer),
                accuracy(pred_class.int(), label_class.int()).detach().numpy())
# show_model_weights(model)