import sys
import torch
import numpy as np
from torch.utils import data
from torch import nn
import pickle
from torch.nn import functional as F
from dataset import GTZAN
from datetime import datetime
from tqdm import tqdm
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

import wandb

def setup_wandb():

    kwargs = {'name': datetime.now().strftime("shallow/%m-%d/%H-%M-%S"), 'project': "ADL-Music-Classication",
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': 'online'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

DEBUG = True

dataset = GTZAN('../data/train_trimmed.pkl')
dataset_val = GTZAN('../data/val_trimmed.pkl')

filename, spectrogram, label, samples = dataset[0] 
class CNN(nn.Module):
    def __init__(self, class_count: int):
        super().__init__()
        self.input_shape = (80, 80)
        self.class_count = class_count

        self.conv1 : nn.Conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding='same'
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 20))
        
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 20),
            padding='same',
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(20, 1))

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.3) # need alpha = 0.3 here.. not sure if that's the same as negative slope
        self.initialise_layer(self.leakyRelu)

        # fully connected map 1x10240 to 200
        self.fc1 = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1)
        # final layer (not specified in paper)
        self.fc2 = nn.Linear(200, self.class_count)
        self.initialise_layer(self.fc2)

    def forward(self, spectograms: torch.Tensor) -> torch.Tensor:
        if DEBUG: print('input shape', spectograms.shape)
        # leaky relu
        x1 = self.conv1(spectograms)
        if DEBUG: print('conv1', x1.shape)
        x1 = F.relu(x1)
        if DEBUG: print('relu1', x1.shape)
        x1 = self.pool1(x1)
        if DEBUG: print('pool1', x1.shape)

        
        x2 = F.relu(self.conv2(spectograms))
        if DEBUG: print('conv2', x2.shape)
        x2 = self.pool2(x2)
        if DEBUG: print('pool2', x2.shape)
        if DEBUG: print((x1.shape, torch.swapaxes(x2, 3, 2).shape))
        
        x = torch.cat((x1, torch.swapaxes(x2, 3, 2)), dim=1)
        if DEBUG: print(x.shape)
        # x = self.flatten(x)
        x = torch.flatten(x, start_dim=1)
        # should have dims 1×10240 after this
        if DEBUG: print(x.shape)
        
        x = self.fc1(x)
        # add leaky relu before dropout
        # x = self.leakyRelu(x)
        x = F.relu(x)
        # add dropout
        x = F.dropout(x, p=0.1)
        # print
        if DEBUG: print('fc1', x.shape)

        x = F.relu(self.fc2(x))
        if DEBUG: print('fc2', x.shape)

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

num_classes = 10
model = CNN(num_classes).to(device)
# # print(spectrogram)
# print("TEST ON BATCH SIZE 1")
# output = model(spectrogram[None, :].to(device))
# print(output)
# # print(spectrogram.dtype)
# spectrograms = [spectrogram[None, :] for filename, spectrogram, label, samples in dataset]
# spectrograms = torch.cat(spectrograms)
# # print(spectrograms)
# # model(torch.Tensor([spectrogram for filename, spectrogram, label, samples in dataset]))
# print("TEST ON BATCH SIZE 10")
# output = model(spectrograms)
# print(output)

# sys.exit()
DEBUG = False
# create batches
dataset = GTZAN('../data/train.pkl')


"""All networks were trained towards
categorical-crossentropy objective using the stochastic
Adam optimization [KB14] with beta1=0.9, beta2=0.999,
epsilon=1e−08 and a learning rate of 0.00005."""

# train model using categorical cross entropy
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
accuracy = Accuracy()

# train model
epochs = 50
val_every_epoch = 1
print('\n Hey \n')
# setup_wandb()
# wandb.init(dir="/Users/zs18656/.config/wandb")
print('aha')
LC_loss, LC_err = [], []
LC_loss_val, LC_err_val = [], []
print('Train Test Dataset Sizes: ', [len(dataset), len(dataset_val)])

for epoch in tqdm(range(epochs)):
    # shuffle dataset
    N = len(dataset)
    N_val = len(dataset_val)
    # generate & shuffle indices
    indices = np.arange(N)
    indices = np.random.permutation(indices)

    losses, losses_val = [], []
    preds, labels = torch.zeros(N), torch.zeros(N)
    preds_val, labels_val = torch.zeros(N_val), torch.zeros(N_val)

    index = 0
    loss = 0
    for filename, spectrogram, label, samples in [dataset[i] for i in indices]:
        # forward pass
        output = model(spectrogram[None, :].to(device))
        # one hot encode label
        label_onehot = nn.functional.one_hot(torch.tensor(label), num_classes=num_classes)[None, :].float().to(device)
        labels[index] = torch.tensor(label)
        # calculate loss
        loss += criterion(output, label_onehot)
        # print(label, loss, output)
        losses.append(loss.cpu().detach().numpy())
        pred_class = torch.argmax(output)
        preds[index] = pred_class
        # l1 normalization
        # l1_norm = torch.norm(model.fc1.weight, p=1)
        # print(l1_norm)
        index += 1
    # print(labels, preds)
    # backpropagate
    loss.backward()
    # update weights
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()

    # LC_loss.append(np.mean(losses))
    LC_loss.append(loss.cpu().detach().numpy()/N)
    LC_err.append(accuracy(preds.int(), labels.int()))
    
    index = 0
    loss = 0
    for filename, spectrogram, label, samples in [dataset_val[i] for i in np.arange(N_val)]:
        output = model(spectrogram[None, :].to(device))
        pred_class = torch.argmax(output)
        preds_val[index] = pred_class
        label_onehot = nn.functional.one_hot(torch.tensor(label), num_classes=num_classes)[None, :].float().to(device)
        labels_val[index] = torch.tensor(label)
        loss += criterion(output, label_onehot)
        losses_val.append(loss.cpu().detach().numpy())
        index += 1
    # LC_loss_val.append(np.mean(losses_val))
    LC_loss_val.append(loss.cpu().detach().numpy()/N_val)
    LC_err_val.append(accuracy(preds_val.int(), labels_val.int()))

    # if epoch % 5 == 0:
    #     fig, ax = plt.subplots(1, 2, figsize = (12,5))
    #     ax[0].hist(preds)
    #     ax[1].hist(preds_val)
    #     plt.show()
# plt.figure()
# plt.plot(LC_err)
# plt.plot(LC_err_val)
# plt.show()
fig, ax = plt.subplots(1, 3, figsize = (12,5))

ax[0].plot(LC_err, label = 'Train Error')
ax[0].plot(LC_err_val, label = 'Test Error')

ax[1].plot(LC_loss, label = 'Train Loss')
ax[1].plot(LC_loss_val, label = 'Test Loss')
ax[1].set_ylim([-0.01,None])

ax[0].legend()
ax[1].legend()
plt.show()