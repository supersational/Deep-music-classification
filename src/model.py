import torch
from torch import nn
from torch.nn import functional as F


class ShallowMusicCNN(nn.Module):
    def __init__(self, class_count: int, disable_half=False):
        super().__init__()
        self.input_shape = (80, 80)
        self.class_count = class_count
        self.disable_half = disable_half

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
        if self.disable_half:
            self.fc1 = nn.Linear(int(10240/2), 200)
        else:
            self.fc1 = nn.Linear(10240, 200)
        self.initialise_layer(self.fc1)
        # final layer (not specified in paper)
        self.fc2 = nn.Linear(200, self.class_count)
        self.initialise_layer(self.fc2)

    def forward(self, spectograms: torch.Tensor, DEBUG=False) -> torch.Tensor:
        if DEBUG: print('input shape', spectograms.shape)
        # leaky relu
        x1 = self.conv1(spectograms)
        if DEBUG: print('conv1', x1.shape)
        x1 = F.relu(x1)
        if DEBUG: print('relu1', x1.shape)
        x1 = self.pool1(x1)
        if DEBUG: print('pool1', x1.shape)

        if self.disable_half:
            x = x1
            
        else:
            
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
        x = F.relu(x)
        # print
        if DEBUG: print('fc1', x.shape)
        x = self.fc2(x)
        """Following typical architecture design, and for consistency with the deep
        architecture/contents of the text, we also add a LeakyReLU with alpha=0.3 after the
        200 unit fully connected layer, before dropout, which is not shown in Figure 1. """
        x = F.leaky_relu(x, negative_slope=0.3)
        if DEBUG: print('fc2', x.shape)
        """Note that the position of Dropout in Figure 1 may cause confusion, the dropout is
        applied AFTER the 200 unit FULLY CONNECTED LAYER as they say in the text, not
        before/after the merge as they show in the figure."""
        x = F.dropout(x, p=0.1)
        # x = F.softmax(x, dim=1)
        # if DEBUG: print('softmax', x.shape)
        # if DEBUG: print('softmax', x)

        # predict the class

        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            # nn.init.xavier_uniform_(layer.weight)
            nn.init.kaiming_normal_(layer.weight)

