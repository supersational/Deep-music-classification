import pickle
try:
    import wget
except ImportError:
    print('wget not installed')
    wget = None

import torch.nn as nn
import numpy as np
import torch
from torch.utils import data
import os

class GTZAN(data.Dataset):
    def __init__(self, dataset_path, download = True):
        """
        Given the dataset path, create the GTZAN dataset. Creates the variable
        self.dataset which is a list of 4-element tuples, each of the form
        (filename, spectrogram, label, samples):
            1) The filename which a given spectrogram belongs to
	        2) The audio spectrogram, which relates to a randomly selected 
                0.93 seconds of audio from "filename". The spectrograms are of 
                size: [1, 80, 80].
	        3) The class/label of the audio file
	        4) The audio samples used to create the spectrogram
        Args:
            dataset_path (str): Path to train.pkl or val.pkl
        """

        train_link = "https://archive.org/download/train_202211/train.pkl"
        val_link   = "https://archive.org/download/train_202211/val.pkl"

        if not os.path.exists(dataset_path) and download:
            working_dir = os.getcwd()
            os.chdir(os.path.dirname(dataset_path))

            _, filename = os.path.split(dataset_path)
            print(filename)
            print(os.getcwd())
            if wget is None:
                raise NotImplementedError(f"{filename} can't be downloaded since wget python package is not installed")
            elif filename == "val.pkl":
                wget.download(val_link)
            elif filename == "train.pkl":
                wget.download(train_link)
            else:
                raise NotImplementedError(f"{filename} is not an available download")

            os.chdir(working_dir)

        self.dataset = pickle.load(open(dataset_path, 'rb'))




    def __getitem__(self, index):
        """
        Given the index from the DataLoader, return the filename, spectrogram, 
        label, and audio samples.
        Args:
            index (int): the dataset index provided by the PyTorch DataLoader.
        Returns:
            filename (str): the filename of the .wav file the spectrogram 
                belongs to.
            spectrogram (torch.Tensor): the audio spectrogram of a 0.93 second
                chunk from the file.
            label (int): the class of the file/spectrogram.
            samples (np.ndarray[np.float64]): the original audio samples /
                amplitude values used to create the given spectrogram
        """
        filename, spectrogram, label, samples = self.dataset[index]
        return filename, spectrogram, label, samples

    def __len__(self):
        """
        Returns the length of the dataset (length of the list of 4-element
            tuples). __len()__ always needs to be defined so that the DataLoader
            can create the batches
        Returns:
            len(self.dataset) (int): the length of the list of 4-element tuples.
        """
        return len(self.dataset)

    def get_batch_indices(self, batch_size, random_seed=42):
        """
        Returns a list of indices that can be used to create batches of data.
        """
        if random_seed:
            np.random.seed(random_seed)

        # shuffle dataset
        N = len(self)

        # generate & shuffle indices
        indices = np.arange(N)
        indices = np.random.permutation(indices)
        index = 0
        while True:
            if index + batch_size > N:
                # shuffle dataset
                indices = np.arange(N)
                indices = np.random.permutation(indices)
                index = 0
            batch_indices = indices[index:index+batch_size]
            index += batch_size
            # print(batch_indices)
            yield batch_indices

    def get_batches(self, batch_size, num_batches=None, random_seed=None):
        """
        Returns a generator that yields batches of data from the dataset.
        """
        batch_num = 0
        num_classes = 10
        batch_indices = self.get_batch_indices(batch_size, random_seed)
        while True:
            batch_data = [self[i] for i in next(batch_indices)]
            spectrograms = torch.stack([spectrogram for filename, spectrogram, label, samples in batch_data])

            # convert label to one hot (cat instead of stack because don't want to add extra dimension)
            label_classes = torch.LongTensor([label % num_classes for filename, spectrogram, label, samples in batch_data])
            labels = nn.functional.one_hot(label_classes, num_classes=num_classes).float()
            yield spectrograms, labels
            batch_num += 1
            if num_batches and batch_num >= num_batches:
                break


def datapoint_to_file(datapoint):
        filename, spectrogram, label, samples = datapoint
        # filename, spectrogram, label, samples = dataset[0]
        print('Filename: {}'.format(filename))
        # decrease the volume of the audio
        samples = samples / (4*1024.0)
        # Write samples to a .wav file
        import scipy.io.wavfile as wav
        wav.write(filename, 22050, samples)
        # Write spectrogram to a .png file
        import matplotlib.pyplot as plt
        plt.imsave(filename+'.png', spectrogram[0], cmap='gray')

# Extensions (recommended)
# - batch norm
# - mixup   - splice two spectrograms together and label them as the average of the two labels

## MR comments
# - extension only have to be 0.1 percent better
# - can submit 'best' version of extension model
# - paper was submitted to a B-class venue
# - if you do five shit extensions that sucks but if you do one really cool extra extension then won't get marked down

if __name__ == '__main__':
    # Set to script directory
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Create the dataset
    setname = 'train'
    dataset = GTZAN('../data/'+setname+'.pkl')
    # Get a random element of the dataset
    # datapoint_to_file(dataset[np.random.randint(0, len(dataset))])

    label_ids = []
    class_labels = {}
    print('starting loop', setname)
    for rep in range(5):
        labels = set()
        while True:
            idx = np.random.randint(0, len(dataset))
            filename, spectrogram, label, samples = dataset[idx]
            if label not in labels:
                labels.add(label)
                label_ids.append(idx)
                class_labels[label] = filename.split('.')[0]
            if len(labels) == 10:
                break
    print(class_labels)
    print(labels)
    print(len(set(label_ids)))
    dataset_trimmed = [dataset[idx] for idx in label_ids]
    pickle.dump(dataset_trimmed, open('../data/'+setname+'_trimmed.pkl', 'wb'))