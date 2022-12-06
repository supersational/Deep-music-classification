import sys
import torch
import numpy as np
from torch import nn
from dataset import GTZAN
from tqdm import tqdm
from datetime import datetime
import argparse


parser = argparse.ArgumentParser(description='Music Classification')
parser.add_argument('--model', type=str, default='deep', help='model to use (deep, shallow, filter)',
    choices=['deep', 'shallow', 'filter'])
parser.add_argument('--batch_size', type=int, default=None, help='batch size')

args = parser.parse_args()


USE_WANDB = False
print(f"{'' if USE_WANDB else 'not'} using wandb")
if USE_WANDB:
    import wandb
from music_classification_models import DeepMusicCNN, ShallowMusicCNN, FilterMusicCNN
from utils import get_batch_ids, plot_losses, plot_accuracies


def setup_wandb(model = "deep", config = {}):

    kwargs = {'name': datetime.now().strftime(f"{model}/%m-%d/%H-%M-%S"), 'project': "ADL-Music-Classication", 'config':config,
            'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': 'online', 'entity':'adl-music-classification'}
    wandb.init(**kwargs)
    wandb.save('*.txt')



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


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
    if args.model == "deep":
        model = DeepMusicCNN(height=height, width=width, channels=1, class_count=10).to(device)
    elif args.model == "shallow":
        model = ShallowMusicCNN(class_count=10).to(device)
    elif args.model == "filter":
        model = FilterMusicCNN(height=height, width=width, channels=1, class_count=10, filter_depth=1/4).to(device)
    else:
        print("invalid model: ", args.model)
        sys.exit(1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, betas=(0.9, 0.999), eps=1e-08)

    n_classes = 10
    epoch_N = 100

    if args.batch_size is not None:
        batch_size = args.batch_size
    elif device == "cuda":
        batch_size = 128
    else:
        batch_size = 16



    config = {"height":height,
                    "width":width,
                    "channels":channels,
                    "lr":lr,
                    "n_epochs":epoch_N,
                    "batch_size":batch_size,
                    "model":"filter"}
    if USE_WANDB:
        setup_wandb(model = "filter_testing", config = config)

    losses, val_losses = [0], [0]
    train_accuracies, val_accuracies = [0], [0]
    val_epochs = [0]
    pbar = tqdm(range(epoch_N))

    for epoch in pbar:
        pbar.set_description(f"Train accuracy: {train_accuracies[-1] if len(train_accuracies) else '0'}")
        class_preds, class_trues = [], []

        for batch_ids in get_batch_ids(N, batch_size):

            spectrograms = torch.stack([spectrogram for filename, spectrogram, label, samples in [dataset[i] for i in batch_ids]])
            label_classes = torch.LongTensor([label for filename, spectrogram, label, samples in [dataset[i] for i in batch_ids]])
            labels = nn.functional.one_hot(label_classes, num_classes=10).float()

            pred = model.forward(spectrograms.to(device))
            batch_loss = criterion(pred, labels.to(device))
            class_preds.extend(torch.argmax(pred, axis=1).cpu().detach())
            class_trues.extend(label_classes)

            batch_loss.backward()
            # update weights
            optimizer.step()
            # zero gradients
            optimizer.zero_grad()
            losses.append(batch_loss.cpu().detach())
        train_success_fail = np.array(class_preds) == np.array(class_trues)
        train_accuracies.append(train_success_fail[train_success_fail].shape[0] / train_success_fail.shape[0])



        #     VALIDATION DATA EVALUATION
        val_loss = 0
        class_preds, val_trues = [], []
        print(f'would be run for {len(get_batch_ids(N_val, batch_size))}x{batch_size} batches but limiting to 10')
        for batch_ids in get_batch_ids(N_val, batch_size)[:10]:
            spectrograms = torch.stack([spectrogram for filename, spectrogram, label, samples in [dataset_val[i] for i in batch_ids]])
            label_classes = torch.LongTensor([label for filename, spectrogram, label, samples in [dataset_val[i] for i in batch_ids]])
            labels = nn.functional.one_hot(label_classes, num_classes=10).float()

            
            with torch.no_grad():
                pred = model.forward(spectrograms.to(device))
            val_loss += criterion(pred, labels.to(device))
            class_preds.extend(torch.argmax(pred, axis=1).cpu().detach())
            val_trues.extend(label_classes)


        val_losses.append(val_loss.cpu().detach())
        val_success_fail = np.array(class_preds) == np.array(val_trues)
        val_accuracies.append(val_success_fail[val_success_fail].shape[0] / val_success_fail.shape[0])


        if USE_WANDB:
            wandb.log({"train_loss":batch_loss.cpu().detach(),
                    "train_acc":train_accuracies[-1],
                    "val_loss":val_loss.cpu().detach(),
                    "val_acc":val_accuracies[-1]})
        elif DEBUG:            print({"train_loss":batch_loss.cpu().detach(),
                    "train_acc":train_accuracies[-1],
                    "val_loss":val_loss.cpu().detach(),
                    "val_acc":val_accuracies[-1]})

    n_batchs = int(N/batch_size)

    plot_accuracies(train_accuracies, val_accuracies)
    plot_losses(losses, losses_val)

    print('final train loss: ', losses[-1])
    print('final test loss: ', val_losses[-1])


