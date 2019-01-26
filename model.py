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
#the model itself, and training it

DATA_PATH = "./data/"

def se_fn(x, amplifying_ratio):
    num_features = x.shape[-1].value
    x = GlobalAvgPool1D()(x)
    x = Reshape((1, num_features))(x)
    x = Dense(num_features * amplifying_ratio, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(num_features, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    return x


def basic_block(x, num_features, weight_decay, _):
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
                kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)
    return x

def rese_block(x, num_features, weight_decay, amplifying_ratio):
    if num_features != x.shape[-1].value:
        shortcut = Conv1D(num_features, kernel_size=1, padding='same', use_bias=True,
                        kernel_regularizer=l2(weight_decay), kernel_initializer='glorot_uniform')(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
                kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(num_features, kernel_size=3, padding='same', use_bias=True,
                kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    if amplifying_ratio > 0:
        x = Multiply()([x, se_fn(x, amplifying_ratio)])
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=3)(x)
    return x

def get_model(block_type='basic', multi=True, init_features=128, amplifying_ratio=16,
                drop_rate=0.5, weight_decay=0., num_classes=12):
    if block_type == 'rese':
        block = rese_block
    elif block_type == 'basic':
        block = basic_block
    else:
        raise Exception('Unknown block type: ' + block_type)

    xc = Input(shape=(59049,1))
    x = Reshape([-1, 1])(xc)

    x = Conv1D(init_features, kernel_size=3, strides=3, padding='valid', use_bias=True,
                kernel_regularizer=l2(weight_decay), kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    num_features = init_features
    layer_outputs = []
    for i in range(9):
        num_features *= 2 if (i == 2 or i == 8) else 1
        x = block(x, num_features, weight_decay, amplifying_ratio)
        layer_outputs.append(x)

    if (multi) and (block_type == 'rese') :
        x = Concatenate()([GlobalMaxPool1D()(output) for output in layer_outputs[-3:]])
    else:
        x = GlobalMaxPool1D()(x)

    x = Dense(x.shape[-1].value, kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if drop_rate > 0.:
        x = Dropout(drop_rate)(x)
    x = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(x)
    model = Model(inputs = xc, outputs = x)
    return model

def generator(x,y,batch_size=10):
    i = 0
    while(True):
        if i+batch_size >= y.shape[0]:
            i = 0
        x_batch = list()
        y_batch = list()
        for j in range(i,i+batch_size):
            x_batch.append(x[j])
            y_batch.append(y[j])
        i = i+batch_size
        x_batch = np.array(x_batch)
        yield x_batch,np.array(y_batch)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
    net_type = input("What model do you want to train? \nType 'basic' for Sample CNN model and 'rese' for the ReSE-2-Multi model.\n")
    
    if net_type == "basic":
        print("Building a Sample CNN model.\n")
    elif net_type == "rese": 
        print("Building a ReSE-2-Multi model.\n")
    else:
        raise Exception('Unknown model type: ' + net_type)

    x_train, x_test, y_train, y_test = prep.get_train_test()
    
    out = get_model(block_type=net_type)
    out.summary()

    out.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.categorical_crossentropy, metrics=['accuracy', f1])

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    batch_size = 10
    steps_per_epoch = y_train.shape[0]//batch_size
    validation_steps = y_test.shape[0]//batch_size
    
    train = generator(x_train,y_train,batch_size=batch_size)
    test = generator(x_test,y_test,batch_size=batch_size)

    if net_type == "basic":
        checkpointer = ModelCheckpoint(filepath='bestModelSCNN.hdf5', verbose=1, save_best_only=True)
    else: 
        checkpointer = ModelCheckpoint(filepath='bestModelReSE.hdf5', verbose=1, save_best_only=True)
    out.fit_generator(train,steps_per_epoch=steps_per_epoch,epochs=10,validation_data=test,validation_steps=validation_steps,callbacks=[checkpointer])

if __name__ == '__main__':
    main()