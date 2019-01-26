import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import scipy 
import scipy.io.wavfile as swave
from skimage.transform import resize
#for preprocessing the data


DATA_PATH = "./data/"

def get_labels(path=DATA_PATH):
    #gets class labels from directory structure
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def wav2array(file_path):
    #converts single .wav file into a numpy array
    rate, data = swave.read(file_path)
    array = resize(data, (59049, 1))
    return array


def save_data_to_array(path=DATA_PATH, max_len=11):
    #saves data into .npy files for training the model
    labels, _, _ = get_labels(path)
    print(labels)
    for label in labels:
        array_vectors = []
        x = 0
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            wavarray = wav2array(wavfile)
            array_vectors.append(wavarray)
            x = x + 1
            if x>600:
                break
        np.save(label + '.npy', array_vectors)


def get_train_test(split_ratio=0.8, random_state=42):
    #split dataset into train and test
    labels, indices, _ = get_labels(DATA_PATH)

    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

if __name__ == '__main__':
  save_data_to_array()
