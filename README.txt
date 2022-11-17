===========
DATA
===========
Training set: train.pkl
Validation/test set: val.pkl

NOTE: In our case, "val" and "test" are used synonymously. In most cases outside of this coursework, there will be three splits train / val / test. It is worth noting the
distinction between val and test. Val is a small subset of the training set where you have access to the ground truth, but you don't train on it and rather use to it to 
tune/optimize hyperparameters of your model. Whereas the test set is used to assess the models ability to generalise to unseen data. You can think of the validation set 
as a "simulation" or "approximation" to the test set.

NOTE: Popular audio processing libraries such as librosa and torchaudio which are used to create log-mel spectrograms are not contained within the BC4 module 
"languages/anaconda3/2019.07-3.6.5-tflow-1.14". If you wish to do the augmentation extension, you will need to create your own, augmented training set locally, and then
reupload that to BC4. *DO NOT TRY TO RUN CODE WHICH RELIES ON LIBROSA OR TORCHAUDIO FOR YOUR AUGMENTATION ON BC4 AS IT WILL NOT FIND THE LIBARY AND WILL CRASH!*

IMPORTANT: If you decide to implement the augmentation-based improvement, you are asked to only upload your augmentation code in your submission. DO *NOT* SUBMIT THE 
AUGMENTED TRAINING SET

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                Data Descriptions									
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The files train.pkl and val.pkl are lists of tuples with four elements:
	1) The filename which a given spectrogram belongs to
	2) The audio spectrogram, which relates to a randomly selected 0.93 seconds of audio from "filename". The spectrograms are of size: [1, 80, 80].
	3) The class/label of the audio file ("blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9)
	4) The audio samples used to create the spectrogram
For the training set (train.pkl) there are 11,750 tuples to train with, for val there are 3,750 tuples to evaluate with. You will use the spectrogram (2) and label (3) 
in order to train and evaluate your model. The filename (1) is provided should you chose to implement the maximum and majority vote metrics as one of your extensions. 
The audio samples (4) are provided should you chose to implement time or pitch shifting as one of your extensions.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                                  The DataLoader									
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
We provide dataset.py, which defines a Pytorch Dataset, which always requires at least the three following functions:
	1) In the __init()__ function, you define how to initialise the data set. In our case, dataset.py will unpickle the dataset provided (either train.pkl or val.pkl) 
	   and store it in self.dataset. 
	2) The __getitem()__ function defines how to retrive an individual sample. In our case, we will return the filename, spectrogram, label and samples according to 
	   the provided index. 
	3) The __len()__ function is always needed to be defined so that the DataLoader knows how to create the batches. In our case, it returns the length of the list 
	   of 4-element tuples.
You can use this dataset by passing it into a torch.utils.data.DataLoader, as shown in the labs. Other functions can be defined (though not necessarily needed for this 
coursework), but the above 3 are *always* required.

==========================================================================================================================================================================
                                                                                        CODE										
==========================================================================================================================================================================

dataset.py: The GTZAN dataset to be passed to the dataloader. It takes as input the path to the training/validation pickle files of the GTZAN dataset (train/val.pkl)

evaluation.py: Code for evaluating the trained model on the validation set. It takes as input two arguments:
    1) --preds: list of the models output for the validation set. This should be a list of the models' outputs (logits) for each spectrogram in 
                the validation set, i.e., A list of length 3750, where each element is a tensor of size 10 (3750 samples in val, 10 classes).
    2) --gts: path to the provided file with the ground truth of the validation set (val.pkl)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                              Additional Information									
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Here we provide some extra detail on how the dataset splits were created, which may assist in your understanding of the data. A better understanding of spectrograms will 
also assist you in the possible data augmentation extension, as you will need to create the spectrograms from the augmented samples yourself.

Spectrograms:
-------------
A spectrogram is an alternative representation of audio. Typically, we are used to audio waveforms, a 1-dimensional signal, which monitors the amplitude of sound over a 
given time period. However, waveforms are rarely used with audio-based deep learning, as they are missing information about the sound signals; the frequencies. To account
for this, a spectrogram is needed. A spectrogram is created from an audio waveform by taking a small chunk (1024 audio samples in our case, referred to as a window size).
The samples within this chunk, are then past through a fourier transform. This provides the amplitude-frequency domain for that small time chunk of audio. This window 
then hops by a number of samples (the hop size, 512 or 50% in our case) and this process is repeated again, so we have amplitude-frequency domains for samples 0-1024, 
and 512-1536. This windowing and hopping is repeated for the entire length of the audio, until you have a series of amplitude-frequency domains for multiple different
timesteps/sample ranges. You then stack these domains in one dimension (time) and you now have a 2D representation of audio, where X=time, Y=Frequency and the value of a 
cell=Amplitude. You can think of spectrograms almost as "single channel audio images", where each column is representing a amplitude-frequency domain at a certain 
timestep. It is important that the hop size < window size, so that there is signal continuity of the audio i.e., you don't miss any frequencies when creating your 
spectrogram. 

Typically, we convert regular spectrograms into log-mel spectrograms. A mel-spectrogram remaps the frequencies to better resemble human hearing. E.g., in a regular 
spectrogram, the space between 2000Hz and 4000Hz will be exactly half as the space between 1000Hz and 2000Hz. In a log-mel spectrogram, these spacings are roughly the 
same to be more comparable with the human perception of sound i.e., humans find it easier to distinguish between similar low frequency sounds rather than similar high 
frequency sounds. 

If you look at images of "audio spectrograms" now, you can see how each column is an amplitude-frequency domain and can even see how certain elements relate to certain
sounds e.g., thin streaks in time with high amplitude/large frequency range (vertical lines) might represent a percussive sound i.e., short and loud sounds. 
Whereas stretched lines in time but with a small frequency range (horizontal lines) might represent a harmonic sound i.e., longer sounds at a consistent 
frequency/amplitude like an instrument holding a note.


How we create the splits:
-------------------------
The dataset provides 1000 .wav audio files, formed by 100 audio clips over 10 classes. All .wav files are 30 seconds in length. To select the spectrograms, the audio is 
divided into chunks of 0.93 seconds, using a step of 50% for each chunk (i.e., each chunk covers seconds [0.00-0.93, 0.47-1.4, 0.93-1.86, ...]) giving 63 chunks per 
.wav file. Of these 63 chunks, 15 are randomly selected per .wav file. These 15 chunks are converted into spectrograms and are used to represent that .wav file. 
Recall there are 1000 .wav files now represented by 15 spectrograms, thus: 1000 * 15 = 15000 = 11750 (size of train) + 3750 (size of val). Train and val are decided by 
selecting 25 random .wav files and their corresponding spectrograms from each class to create the val set ((25 * 10) * 15 = 3750) and using the remaining spectrograms
to create the train set ((75 * 10) * 15 = 11750). Thus the train and test set are perfectly balanced (as all things should be).

NOTE: The above paragraph talks about data split creation and is not how datasets are described in papers. Thus, you should *NOT* try to describe it in this way for the 
dataset section in your report.


