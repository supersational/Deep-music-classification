import torch
import numpy as np
from torch.utils import data
import pickle

class GTZAN(data.Dataset):
    def __init__(self, dataset_path):
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
    dataset = GTZAN('../data/train.pkl')
    # Get a random element of the dataset
    datapoint_to_file(dataset[np.random.randint(0, len(dataset))])
    