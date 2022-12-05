import sys
import torch
import numpy as np
from torch import nn
from dataset import GTZAN
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
from music_classification_models import DeepMusicCNN, ShallowMusicCNN, FilterMusicCNN
from utils import get_batch_ids, plot_losses, plot_accuracies


def setup_wandb(model = "deep"):

    kwargs = {'name': datetime.now().strftime(f"{model}/%m-%d/%H-%M-%S"), 'project': "ADL-Music-Classication",
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': 'online', 'entity':'adl-music-classification'}
    wandb.init(**kwargs)
    wandb.save('*.txt')



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    setup_wandb(model = "filter_testing")

    print(f"Running with device: {device}")
    DEBUG = True

    dataset = GTZAN('../data/train.pkl')
    dataset_val = GTZAN('../data/val.pkl')

    N, N_val = len(dataset), len(dataset_val)
    filename, spectrogram, label, samples = dataset[0]

    print('Training Data Size: ', N)
    print('Testing Data Size: ', N_val)


    height, width, channels = 80, 80, 1
    lr = 0.001
    model = FilterMusicCNN(height=height, width=width, channels=1, class_count=10, filter_depth=1/3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999), eps=1e-08)
    losses, losses_val = [], []
    n_classes = 10
    epoch_N = 300

    if device == "cuda":
        batch_size = 128
    else:
        batch_size = 16

    wandb.config = {"height":height,
                    "width":width,
                    "channels":channels,
                    "lr":lr,
                    "n_epochs":epoch_N,
                    "batch_size":batch_size}

    losses, val_losses = [], []
    train_accuracies, val_accuracies = [0], [0]
    pbar = tqdm(range(epoch_N))

    for epoch in pbar:
        pbar.set_description(f"Train accuracy: {train_accuracies[-1]}")
        class_preds, class_trues = [], []

        for batch_ids in get_batch_ids(N, batch_size):
            batch_loss = 0
            for i in batch_ids:
                label_i_tensor = torch.zeros(n_classes)
                filename_i, spectrogram_i, label_i, samples_i = dataset[i]
                label_i_tensor[label_i] = 1
                pred = model.forward(spectrogram_i.to(device))
                batch_loss += criterion(pred, label_i_tensor.to(device))

                class_preds.append(torch.argmax(pred).cpu().detach())
                class_trues.append(label_i)

            batch_loss.backward()
            # update weights
            optimizer.step()
            # zero gradients
            optimizer.zero_grad()
            losses.append(batch_loss.cpu().detach())
        train_success_fail = np.array(class_preds) == np.array(class_trues)
        train_accuracies.append(train_success_fail[train_success_fail].shape[0] / train_success_fail.shape[0])

        wandb.log({"train_loss":batch_loss.cpu().detach(),
                   "train_acc":train_success_fail[train_success_fail].shape[0] / train_success_fail.shape[0]})

        #     VALIDATION DATA EVALUATION
        val_loss = 0
        class_preds, val_trues = [], []
        for i in range(N_val):
            label_i_tensor = torch.zeros(n_classes)
            filename_i, spectrogram_i, label_i, samples_i = dataset_val[i]
            label_i_tensor[label_i] = 1
            with torch.no_grad():
                pred = model.forward(spectrogram_i.to(device))
            val_loss += criterion(pred, label_i_tensor.to(device))

            class_preds.append(torch.argmax(pred).cpu().detach())
            val_trues.append(label_i)

        val_losses.append(val_loss.cpu().detach())
        val_success_fail = np.array(class_preds) == np.array(val_trues)
        val_accuracies.append(val_success_fail[val_success_fail].shape[0] / val_success_fail.shape[0])
        wandb.log({"val_loss":val_loss.cpu().detach(),
                   "val_acc":val_success_fail[val_success_fail].shape[0] / val_success_fail.shape[0]})

    n_batchs = int(N/batch_size)

    plot_accuracies(train_accuracies, val_accuracies)
    plot_losses(losses, losses_val)

    print('final train loss: ', losses[-1])
    print('final test loss: ', val_losses[-1])


