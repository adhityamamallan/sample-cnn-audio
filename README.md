# Keras implementation of the paper "Raw Waveform-based Audio Classification Using Sample-level CNN Architectures"

#### Note: This model requires the Tensorflow Speech Recognition Dataset (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/leaderboard) to train. 

### Environment Setup: 

OS : Ubuntu 18.04.1 LTS

Language : Python 3.6.6

Environment : Anaconda 4.5.11

### How to run this model:

#### 1)  Run prep.py to preprocess the dataset and convert it into a set of .npy files.

    $ python prep.py

The .npy files will appear in the base directory along with the other .py scripts.

#### 2a) Run model.py to set up and train the model.
    
    $ python model.py

When you do so, you will be prompted to pick the type of model to build and train.
After you do so, a summary of all the layers in the model will appear in the
terminal window, and then training will begin.

Training the SampleCNN took ~15 minutes, and training the ReSE-2-Multi took
~35 minutes, for 10 epochs, on a batch size of ~550, using an Nvidia GTX 1060.

After the completion of training, the script will save the trained model as an 
.hdf5 file in the base directory.
    
#### 2b) Alternatively, run one of the pretrained models using evaluate.py.
    
    $ python evaluate.py

You will be prompted to pick the model you want to evaluate. 
After you do so, a summary of the model will appear, and the weights file 
is loaded. You do not need to specify the name of the weights (.hdf5) file; the script
automatically picks the right one based on what model you choose to evaluate.

Of course, if the .hdf5 file is missing, an exception is thrown.

 After evaluation, the loss, accuracy and F1-score are shown.

### Libraries used :

#### Keras 2.2.4

A neural network library acting as a wrapper for Tensorflow-GPU

Used for modelling and manipulating CNNs.
    
#### Tensorflow-GPU 1.11.0

An open source neural network library

Backend for Keras.
    
#### NumPy 1.15.4

A library for scientific computing
    
Used for arrays and related functionality.
    
#### SciPy 1.1.0

An extension of NumPy

scipy.io.wavfile.read is used for reading .wav file data into a NumPy array.
    
#### scikit-learn

A collection of tools for data mining and data analysis

sklearn.model_selection.train_test_split is used for splitting the dataset into training set and validation set.
    
#### scikit-image

A collection of tools for image processing

skimage.transform.resize is used for converting the read audio file 
(in NumPy array) into a NumPy array of size 59049 for the model to use.

#### tqdm

A progress bar addon

Shows progress of writing .wav files to a .npy file.
