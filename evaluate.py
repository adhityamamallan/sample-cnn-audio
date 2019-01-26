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

DATA_PATH = "./data/"

def main():
    net_type = input("What model do you want to evaluate? \nType 'basic' for Sample CNN model and 'rese' for the ReSE-2-Multi model.\n")
    
    if net_type == "basic":
        net_wts = "bestModelSCNN.hdf5"
    elif net_type == "rese": 
        net_wts = "bestModelReSE.hdf5"
    else:
        raise Exception('Pretrained model not found: ' + net_type)

    x_train, x_test, y_train, y_test = prep.get_train_test()
    
    out = model.get_model(block_type=net_type)
    out.summary()
    out.load_weights(net_wts)
    out.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics=['accuracy', model.f1])

    y_test = to_categorical(y_test)

    batch_size = 10
    validation_steps = y_test.shape[0]//batch_size
    
    test = model.generator(x_test,y_test,batch_size=batch_size)

    loss, accuracy, f1 = out.evaluate_generator(test, steps = validation_steps)

    print("Loss = " + str(loss))
    print("Accuracy = " + str(accuracy*100) + "%")
    print("F1 score = " + str(f1))

if __name__ == '__main__':
    main()