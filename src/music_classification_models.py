import torch
from torch import nn
from torch.nn import functional as F

def initialise_layer(layer):
    if hasattr(layer, "bias"):
        nn.init.zeros_(layer.bias)
    if hasattr(layer, "weight"):
        # nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_normal_(layer.weight)


class ShallowMusicCNN(nn.Module):
    def __init__(self, class_count: int, dropout: float = 0.1, alpha: float = 0.3 ):
        super().__init__()
        self.class_count = class_count
        self.dropout = dropout
        self.alpha = alpha

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(10, 23),
            padding='same'
        )

        initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 20))

        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(21, 20),
            padding='same',
        )
        initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(20, 1))

        # fully connected map 1x10240 to 200
        self.fc1 = nn.Linear(10240, 200)
        initialise_layer(self.fc1)

        # final layer (not specified in paper)
        self.fc2 = nn.Linear(200, self.class_count)
        initialise_layer(self.fc2)

    def forward(self, spectograms: torch.Tensor) -> torch.Tensor:

        x1 = self.conv1(spectograms)
        x1 = F.relu(x1)
        x1 = self.pool1(x1)


        x2 = F.relu(self.conv2(spectograms))
        x2 = self.pool2(x2)

        x = torch.cat((x1, torch.swapaxes(x2, 3, 2)), dim=1)

        x = torch.flatten(x, start_dim=1)
        # should have dims 1×10240 after this

        x = self.fc1(x)
        if self.alpha is not None:
            x = F.leaky_relu(x, negative_slope=self.alpha)
        else:
            x = F.relu(x)

        x = self.fc2(x)
        """Following typical architecture design, and for consistency with the deep
        architecture/contents of the text, we also add a alpha with alpha=0.3 after the
        200 unit fully connected layer, before dropout, which is not shown in Figure 1. """

        """Note that the position of Dropout in Figure 1 may cause confusion, the dropout is
        applied AFTER the 200 unit FULLY CONNECTED LAYER as they say in the text, not
        before/after the merge as they show in the figure."""
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout)

        return x



class DeepMusicCNN(nn.Module):
    def __init__(self, class_count: int, dropout: float = 0.25, alpha: float = 0.3 ):
        super().__init__()
        self.class_count = class_count
        self.dropout = dropout
        self.alpha = alpha

        self.conv11 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(10, 23),
            padding='same')
        self.conv12 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(21, 20),
            padding='same')

        initialise_layer(self.conv11)
        self.pool11 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        initialise_layer(self.conv12)
        self.pool12 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv21 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 11),
            padding='same')
        self.conv22 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(10, 5),
            padding='same')

        initialise_layer(self.conv21)
        self.pool21 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        initialise_layer(self.conv22)
        self.pool22 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv31 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 5),
            padding='same')
        self.conv32 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 3),
            padding='same')

        initialise_layer(self.conv31)
        self.pool31 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        initialise_layer(self.conv32)
        self.pool32 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv41 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 4),
            padding='same')
        self.conv42 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(4, 2),
            padding='same')

        initialise_layer(self.conv41)
        self.pool41 = nn.MaxPool2d(kernel_size=(1, 5))  # Check stride
        initialise_layer(self.conv42)
        self.pool42 = nn.MaxPool2d(kernel_size=(5, 1))

        self.fc1 = nn.Linear(5120, 200)
        initialise_layer(self.fc1)
        self.fc2 = nn.Linear(200, 10)
        initialise_layer(self.fc2)

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
        x_right = torch.flatten(x_right, start_dim=1)

        x_conc = torch.cat((x_left, x_right), dim=1)

        x_conc = self.fc1(x_conc)
        if self.alpha is not None:
            x_conc = F.leaky_relu(x_conc, negative_slope=self.alpha)
        else:
            x_conc = F.relu(x_conc)

        x_conc = self.fc2(x_conc)

        if self.dropout is not None:
            x_conc = F.dropout(x_conc, p=self.dropout)

        return x_conc


class FilterMusicCNN(nn.Module):
    "3-branch model with each branch applying a different range of frequency-filter"
    def __init__(self, class_count: int, filter_depth: float, dropout: float = 0.25, alpha: float = 0.3):
        super().__init__()
        self.class_count = class_count
        self.dropout = dropout
        self.alpha = alpha
        height = 80
        self.filter_dim  = int(height * filter_depth)

        self.convhigh1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(10, 23),
            padding='same')
        self.convmid1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(21, 20),
            padding='same')
        self.convlow1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(21, 20),
            padding='same')

        initialise_layer(self.convhigh1)
        self.poolhigh1 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        initialise_layer(self.convmid1)
        self.poolmid1 = nn.MaxPool2d(kernel_size=(2, 2))
        initialise_layer(self.convlow1)
        self.poollow1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        #================================
        
        self.convhigh2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 11),
            padding='same')
        self.convmid2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 11),
            padding='same')
        self.convlow2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(5, 11),
            padding='same')

        initialise_layer(self.convhigh2)
        self.poolhigh2 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        initialise_layer(self.convmid2)
        self.poolmid2 = nn.MaxPool2d(kernel_size=(2, 2))
        initialise_layer(self.convlow2)
        self.poollow2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        #==========================
        
        self.convhigh3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 5),
            padding='same')
        self.convmid3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 5),
            padding='same')
        self.convlow3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 5),
            padding='same')

        initialise_layer(self.convhigh3)
        self.poolhigh3 = nn.MaxPool2d(kernel_size=(2, 2))  # Check stride
        initialise_layer(self.convmid3)
        self.poolmid3 = nn.MaxPool2d(kernel_size=(2, 2))
        initialise_layer(self.convlow3)
        self.poollow3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        #==========================
        
        self.convhigh4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 4),
            padding='same')
        self.convmid4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 4),
            padding='same')
        self.convlow4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(2, 4),
            padding='same')

        initialise_layer(self.convhigh4)
        self.poolhigh4 = nn.MaxPool2d(kernel_size=(1, 5))  # Check stride
        initialise_layer(self.convmid4)
        self.poolmid4 = nn.MaxPool2d(kernel_size=(1, 5))
        initialise_layer(self.convlow4)
        self.poollow4 = nn.MaxPool2d(kernel_size=(1, 5))
        
        
        #==========================
        # self.fc1 = nn.Linear(5120, 200)
        self.fc1 = nn.Linear(7680, 200)
        initialise_layer(self.fc1)
        self.fc2 = nn.Linear(200, 10)
        initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        x_low  = torch.clone(images)
        x_low[:, :self.filter_dim, :] = 0.

        x_mid  = torch.clone(images)
        x_mid[:, self.filter_dim:-self.filter_dim, :] = 0.

        x_high = torch.clone(images)
        x_high[:, :-self.filter_dim, :] = 0

        x_low, x_mid, x_high = F.relu(self.convlow1(x_low)), F.relu(self.convmid1(x_mid)), F.relu(self.convhigh1(x_high))
        x_low, x_mid, x_high = self.poollow1(x_low), self.poolmid1(x_mid), self.poolhigh1(x_high)

        x_low, x_mid, x_high = F.relu(self.convlow2(x_low)), F.relu(self.convmid2(x_mid)), F.relu(self.convhigh2(x_high))
        x_low, x_mid, x_high = self.poollow2(x_low), self.poolmid2(x_mid), self.poolhigh2(x_high)

        x_low, x_mid, x_high = F.relu(self.convlow3(x_low)), F.relu(self.convmid3(x_mid)), F.relu(self.convhigh3(x_high))
        x_low, x_mid, x_high = self.poollow3(x_low), self.poolmid3(x_mid), self.poolhigh3(x_high)
     
        x_low, x_mid, x_high = F.relu(self.convlow4(x_low)), F.relu(self.convmid4(x_mid)), F.relu(self.convhigh4(x_high))
        x_low, x_mid, x_high = self.poollow4(x_low), self.poolmid4(x_mid), self.poolhigh4(x_high)


        x_low = torch.flatten(x_low, start_dim=1)

        x_mid = torch.flatten(x_mid, start_dim=1)

        x_high = torch.flatten(x_high, start_dim=1)

        x_conc = torch.cat((x_low, x_mid, x_high), dim=1)
        x_conc = self.fc1(x_conc)
        if self.alpha is not None:
            x_conc = F.leaky_relu(x_conc, negative_slope=self.alpha)
        else:
            x_conc = F.relu(x_conc)

        x_conc = self.fc2(x_conc)
        
        if self.dropout is not None:
            x_conc = F.dropout(x_conc, p=self.dropout)
        return x_conc
