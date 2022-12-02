import numpy as np
import random
import matplotlib.pyplot as plt

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

def plot_losses(train_loss, val_loss):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axes[0].plot((train_loss), label='train')
    axes[1].plot((val_loss), label='test')
    axes[0].legend()
    axes[1].legend()

    plt.savefig('../results/losses.png')
    plt.close()

def plot_accuracies(train_accuracies, val_accuracies):
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    print('Train Acc: ', train_accuracies[-1])
    print('Test Acc: ', val_accuracies[-1])

    plt.savefig('../results/accuracies.png')
    plt.close()