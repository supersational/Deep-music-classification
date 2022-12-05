import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, tag = ''):
    fig, ax = plt.subplots(figsize = (6,6))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax = ax)

    plt.tight_layout()
    plt.savefig(f'../results/confusion{tag}.png')

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

def plot_losses(train_loss, val_loss, tag = ''):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot((train_loss), label='train')
    axes[1].plot((val_loss), label='test')
    axes[0].legend()
    axes[1].legend()

    plt.savefig(f'../results/losses{tag}.png')
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies, tag = ''):
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    print('Train Acc: ', train_accuracies[-1])
    print('Test Acc: ', val_accuracies[-1])

    plt.savefig(f'../results/accuracies{tag}.png')
    plt.close()