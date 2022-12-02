import torch
from torch import nn
from torch.nn import functional as F


class ShallowMusicCNN(nn.Module):
    def __init__(self, class_count: int, disable_half=False):
        super().__init__()
        self.input_shape = (80, 80)
        self.class_count = class_count
        self.disable_half = disable_half

        self.conv1: nn.Conv2d = nn.Conv2d(
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

        self.leakyRelu = nn.LeakyReLU(
            negative_slope=0.3)  # need alpha = 0.3 here.. not sure if that's the same as negative slope
        self.initialise_layer(self.leakyRelu)

        # fully connected map 1x10240 to 200
        if self.disable_half:
            self.fc1 = nn.Linear(int(10240 / 2), 200)
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


class DeepMusicCNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = (height, width, channels)
        self.class_count = class_count

        self.conv11 = nn.Conv2d(
            in_channels=self.input_shape[2], out_channels=16, kernel_size=(10, 23),
            padding='same')
        self.conv12 = nn.Conv2d(
            in_channels=self.input_shape[2], out_channels=16, kernel_size=(21, 20),
            padding='same')

        self.initialise_layer(self.conv11)
        self.pool11 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        self.initialise_layer(self.conv12)
        self.pool12 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv21 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 11),
            padding='same')
        self.conv22 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(10, 5),
            padding='same')

        self.initialise_layer(self.conv21)
        self.pool21 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        self.initialise_layer(self.conv22)
        self.pool22 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv31 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 5),
            padding='same')
        self.conv32 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 3),
            padding='same')

        self.initialise_layer(self.conv31)
        self.pool31 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        self.initialise_layer(self.conv32)
        self.pool32 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv41 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 4),
            padding='same')
        self.conv42 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(4, 2),
            padding='same')

        self.initialise_layer(self.conv41)
        self.pool41 = nn.MaxPool2d(kernel_size=(1, 5))  # Check stride
        self.initialise_layer(self.conv42)
        self.pool42 = nn.MaxPool2d(kernel_size=(5, 1))

        self.fc1 = nn.Linear(5120, 200)
        self.initialise_layer(self.fc1)
        self.fc2 = nn.Linear(200, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x_left, x_right = F.relu(self.conv11(images)), F.relu(self.conv12(images))
        x_left, x_right = self.pool11(x_left), self.pool12(x_right)

        x_left, x_right = F.relu(self.conv21(x_left)), F.relu(self.conv22(x_right))
        x_left, x_right = self.pool21(x_left), self.pool22(x_right)

        x_left, x_right = F.relu(self.conv31(x_left)), F.relu(self.conv32(x_right))
        x_left, x_right = self.pool31(x_left), self.pool32(x_right)

        x_left, x_right = F.relu(self.conv41(x_left)), F.relu(self.conv42(x_right))
        x_left, x_right = self.pool41(x_left), self.pool42(x_right)

        x_left = torch.flatten(x_left, start_dim=1)
        x_left = torch.flatten(x_left)

        x_right = torch.flatten(x_right, start_dim=1)
        x_right = torch.flatten(x_right)

        x_conc = torch.cat((x_left, x_right))
        x_conc = F.relu(self.fc1(x_conc))
        x_conc = self.fc2(x_conc)
        return x_conc

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


