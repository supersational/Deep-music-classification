import torch
from dataset import GTZAN
import sys
# import torch.nn.functional as F
import torch.nn as nn

from model import ShallowMusicCNN
dataset = GTZAN('../data/train.pkl')
filename, spectrogram, label, samples = dataset[0] 



num_classes = 2
model = ShallowMusicCNN(num_classes)
# model


# def show_model_weights(model):
#     layers = list(model.modules())[0]
#     layers = model.modules()
#     print(layers)
#     for layer in layers:
#         print('='*20)
#         print(type(layer), layer)
#         if hasattr(layer, "weight"):
#             print(layer.weight)
#         else:
#             print('no weight')



# spectrograms = torch.stack([spectrogram for filename, spectrogram, label, samples in dataset])

# # convert label to one hot (cat instead of stack because don't want to add extra dimension)
# label_class = torch.LongTensor([label % num_classes for filename, spectrogram, label, samples in dataset])
# labels = nn.functional.one_hot(label_class, num_classes=num_classes).float()


import numpy as np
from torchmetrics import Accuracy



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
accuracy = Accuracy()
batcher = dataset.get_batches(batch_size=20, num_batches=None, random_seed=42)

# for epoch in range(1000):
for epoch, (spectrograms, labels) in enumerate(batcher):

    output = model(spectrograms)
    if epoch == 0:
        print(output)
    preds = output
    pred_class = preds.argmax(dim=1)
    label_class = labels.argmax(dim=1)
    # print(pred_class)
    loss = criterion(preds, labels)

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
        model.fc1.weight[0, 0:4].detach().numpy(),
        model.fc2.weight[0, 0:4].detach().numpy(),
        model.conv1.weight[0, 0, 0, 0:1].detach().numpy(),
        model.conv2.weight[0, 0, 0, 0:1].detach().numpy(),
    )
    print(epoch, accuracy(pred_class.int(), label_class.int()).detach().numpy(), torch.round(loss,decimals=3).detach().numpy(), torch.round(preds.sum(dim=0),decimals=3).detach().numpy(),*wvis)
# show_model_weights(model)