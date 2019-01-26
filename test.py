import os
import keras
from keras import backend as K
from keras.models import (Sequential, Model)
from keras.regularizers import l2
from keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply, GlobalMaxPool1D,
                          Dense, Dropout, Activation, Reshape, Input, Concatenate, Add)
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import scipy 
import scipy.io.wavfile as swave
from skimage.transform import resize
import prep
import tensorflow as tf
import model
#for testing the model


DATA_PATH = "./data/"

def predtoclass(number):
    #returns the class label when given class number
    labels, _, _ = prep.get_labels(DATA_PATH)
    return labels[number]

def main():
    #needs pretrained models for testing
    list_files = []
    list_fnames = []
    while(1):
        #make list of files for testing
        fname = input("Enter the files you want to process; enter 'end' when you want to stop.\n")
        if fname == "end":
            break
        else:
            files = prep.wav2array("/home/adhitya/Desktop/NNFL/test/" + fname)
            list_files.append(files)
            list_fnames.append(fname)
    array_files = np.array(list_files)
    net_type = input("What model do you want to evaluate? \nType 'basic' for Sample CNN model and 'rese' for the ReSE-2-Multi model.\n")
    
    if net_type == "basic":
        net_wts = "bestModelSCNN.hdf5"
    elif net_type == "rese": 
        net_wts = "bestModelReSE.hdf5"
    else:
        raise Exception('Pretrained model not found: ' + net_type)
    
    out = model.get_model(block_type=net_type)
    out.summary()
    out.load_weights(net_wts)
    out.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics=['accuracy', model.f1])

    pred = out.predict(array_files, batch_size=2, verbose=0, steps=None)

    for i in range(len(list_fnames)):
        #print filename with predicted output
        print((list_fnames[i]) + ': ' + predtoclass(np.argmax(pred[i])))

if __name__ == '__main__':
    main()
