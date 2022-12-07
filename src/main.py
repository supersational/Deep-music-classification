import os
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
parser.add_argument('--tag', type=str, default='', help='tag for saving results')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--run_n', type=int, default=None, help='run id to do multiple runs')
parser.add_argument('--l1', type=float, default=0.001, help='l1 regularization')
parser.add_argument('--dropout', type=float, default=None, help='dropout rate, paper uses 0.1 for shallow and 0.25 for deep')
parser.add_argument('--alpha', type=float, default=None, help='alpha for leaky relu, 0 for relu')
parser.add_argument('--wandb', action='store_true', help='use wandb')
args = parser.parse_args()

tag = args.model+('_'+args.tag if args.tag else '')+('_'+str(args.run_n) if args.run_n else '')


print(f"{'' if args.wandb else 'not'} using wandb")
if args.wandb:
    import wandb
from music_classification_models import DeepMusicCNN, ShallowMusicCNN, FilterMusicCNN
from utils import get_batch_ids, plot_losses, plot_accuracies, plot_confusion_matrix


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
    class_names = {8: 'reggae', 1: 'classical', 6: 'metal', 2: 'country', 5: 'jazz', 7: 'pop', 4: 'hiphop', 0: 'blues', 3: 'disco', 9: 'rock'}

    N, N_val = len(dataset), len(dataset_val)
    filename, spectrogram, label, samples = dataset[0]

    print('Training Data Size: ', N)
    print('Testing Data Size: ', N_val)


    height, width, channels = 80, 80, 1
    
    """L1 weight
    regularization with a penalty of 0.0001 was applied to all
    trainable parameters"""

    lr = 0.00005
    l1_lambda = args.l1
    model_args = {"class_count":10, "alpha": args.alpha, "dropout": args.dropout}
    # ensure dropout defaults to values given in paper
    if args.dropout is None:
        if args.model == 'shallow':
            model_args["dropout"] = 0.1
        else:
            model_args["dropout"] = 0.25

    if args.alpha is None:
        model_args["alpha"] = 0.3
    

    if args.model == "deep":
        model = DeepMusicCNN(**model_args).to(device)
    elif args.model == "shallow":
        model = ShallowMusicCNN(**model_args).to(device)
    elif args.model == "filter":
        model = FilterMusicCNN(**model_args, filter_depth=1/3, device = device).to(device)
    else:
        print("invalid model: ", args.model)
        sys.exit(1)

    ## after using the model param add the run_n to not overwrite other runs
    if args.run_n is not None:
        args.model += '_'+str(args.run_n)
    os.makedirs(f'../results/{args.model}', exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    
    n_classes = 10
    epoch_N = args.epochs

    if args.batch_size is not None:
        batch_size = args.batch_size
    elif device == "cuda":
        batch_size = 128
    else:
        batch_size = 16



    config = {  "model_name":args.model,
                "height":height,
                "width":width,
                "channels":channels,
                "lr":lr,
                "l1_lambda":l1_lambda,
                "n_epochs":epoch_N,
                "batch_size":batch_size,
                "model":"filter"}
    if args.wandb:
        setup_wandb(model = tag, config = config)

    losses, val_losses = [0], [0]
    train_accuracies, val_accuracies = [0], [0]
    val_epochs = [0]
    pbar = tqdm(range(1,epoch_N+1))

    for epoch in pbar:
        pbar.set_description(f"Train accuracy: {train_accuracies[-1]:.2f}")
        class_preds, class_trues = [], []

        for batch_ids in get_batch_ids(N, batch_size):

            spectrograms = torch.stack([spectrogram for filename, spectrogram, label, samples in [dataset[i] for i in batch_ids]])
            label_classes = torch.LongTensor([label for filename, spectrogram, label, samples in [dataset[i] for i in batch_ids]])
            labels = nn.functional.one_hot(label_classes, num_classes=10).float()

            pred = model.forward(spectrograms.to(device))
            

            batch_loss = criterion(pred, labels.to(device))
            if l1_lambda > 0:
                weights = torch.cat([p.view(-1) for n, p in model.named_parameters() if ".weight" in n])
                batch_loss += l1_lambda * torch.norm(weights, 1)
                
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

        if args.wandb:
            wandb.log({"train_loss":batch_loss.cpu().detach(),
                    "train_acc":train_accuracies[-1]}, 
                    step=epoch, commit=False)

        # every 10 epochs, evaluate on validation data (and on final epoch)
        if (epoch % 10) == 0:
        
            #     VALIDATION DATA EVALUATION
            val_loss = 0

            val_preds, val_trues = [], []
            for batch_ids in get_batch_ids(N_val, batch_size):
                spectrograms = torch.stack([spectrogram for filename, spectrogram, label, samples in [dataset_val[i] for i in batch_ids]])
                label_classes = torch.LongTensor([label for filename, spectrogram, label, samples in [dataset_val[i] for i in batch_ids]])
                labels = nn.functional.one_hot(label_classes, num_classes=10).float()

                
                with torch.no_grad():
                    pred = model.forward(spectrograms.to(device))
                val_loss += criterion(pred, labels.to(device))

                if l1_lambda > 0:
                    weights = torch.cat([p.view(-1) for n, p in model.named_parameters() if ".weight" in n])
                    val_loss += l1_lambda * torch.norm(weights, 1)
                
                val_preds.extend(torch.argmax(pred, axis=1).cpu().detach())
                val_trues.extend(label_classes)

            val_epochs.append(epoch)
            val_losses.append(val_loss.cpu().detach())
            val_success_fail = np.array(val_preds) == np.array(val_trues)
            val_accuracies.append(val_success_fail[val_success_fail].shape[0] / val_success_fail.shape[0])
            # every 100 epochs write results
            if (epoch % 100) == 0:


                if args.wandb:
                    wandb.log({"train_loss":batch_loss.cpu().detach(),
                            "train_acc":train_accuracies[-1],
                            "val_loss":val_loss.cpu().detach(),
                            "val_acc":val_accuracies[-1]}, step=epoch)
                elif DEBUG: 
                        print({"train_loss":batch_loss.cpu().detach(),
                        "train_acc":train_accuracies[-1],
                        "val_loss":val_loss.cpu().detach(),
                        "val_acc":val_accuracies[-1]})

                        plot_accuracies(train_accuracies, val_accuracies, val_epochs,
                                        tag=f'_{tag}_{epoch}',
                                        title=f'{args.model.title()} model\n Accuracy: {val_accuracies[-1]:.2f}',
                                        model = args.model)

                        plot_losses(losses, val_losses, val_epochs,
                                    tag=f'_{tag}_{epoch}',
                                    title=f'{args.model.title()} model',
                                    model = args.model)

                        plot_confusion_matrix(np.array(val_preds), np.array(val_trues),
                                            tag=f'_{tag}_{epoch}',
                                            model = args.model)
                        if args.wandb: wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                                                y_true=np.array(val_trues), preds=np.array(val_preds),
                                                class_names=class_names)})

    with open(f'../results/{args.model}/results_{tag}.txt') as f:
        f.write(f"""
final train loss: {losses[-1]}
final test loss: {val_losses[-1]}""")

    print('final train loss: ', losses[-1])
    print('final test loss: ', val_losses[-1])

    np.save(f'../results/{args.model}/train_accuracies_{tag}.npy', train_accuracies)
    np.save(f'../results/{args.model}/val_accuracies_{tag}.npy', val_accuracies)
    np.save(f'../results/{args.model}/val_epochs_{tag}.npy', val_epochs)
    np.save(f'../results/{args.model}/losses_{tag}.npy', losses)
    np.save(f'../results/{args.model}/val_losses_{tag}.npy', val_losses)

    # plot_confusion_matrix(np.array(val_preds), np.array(val_trues),
    #                       tag=f'_{tag}',
    #                       model = args.model)
