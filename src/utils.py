"""
Riku Green, Sven Hollowell, Alex Davies

Utility functions for plotting and batch indices
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, tag = '', model = 'nomodel'):
    """Plot a confusion matrix based on real and predicted class labels"""
    fig, ax = plt.subplots(figsize = (6,6))


    # BC4 version of sklearn has this api
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)

    plt.tight_layout()
    plt.savefig(f'../results/{model}/confusion{tag}.png')
    plt.close()


def get_batch_ids(N, batch_size):
    sample_ids = np.arange(N)
    random.shuffle(sample_ids)
    num_batchs = N / batch_size

    sampled_n = 0
    batch_ids = []
    for batch in range(int(num_batchs)):
        batch_i = sample_ids[batch * batch_size:(batch + 1) * batch_size]
        sampled_n += batch_size
        batch_ids.append(batch_i)
    if num_batchs != int(num_batchs):
        batch_i = (sample_ids[sampled_n:])
        batch_ids.append(batch_i)
    return batch_ids

def plot_losses(train_loss, val_loss, val_epochs, tag = '', title='', model = 'nomodel'):
    """Plot training and validation losses"""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot((train_loss), label='train')
    axes[1].plot(val_epochs, val_loss, label='test')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_title('Train Loss')
    axes[1].set_title('Test Loss')
    if title: 
        fig.suptitle(title)


    os.makedirs('../results', exist_ok=True)
    plt.savefig(f'../results/{model}/losses{tag}.png')
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies, val_epochs, tag = '', title='', model = 'nomodel'):
    """Plot training and validation accuracies"""
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_epochs, val_accuracies, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    if title: plt.title(title)
    print('Train Acc: ', train_accuracies[-1])
    print('Test Acc: ', val_accuracies[-1])

    os.makedirs('../results', exist_ok=True)
    plt.savefig(f'../results/{model}/accuracies{tag}.png')
    plt.close()