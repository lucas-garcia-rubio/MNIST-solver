#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:55:59 2020

@author: lucasgr

Agora é só experimentar diferentes parâmetros no modelo
"""

# Digit Recognizer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%%

train = pd.read_csv(r"/home/lucasgr/Área de trabalho/digit-recognizer/train.csv")
test = pd.read_csv(r"/home/lucasgr/Área de trabalho/digit-recognizer/test.csv")


#%% Splitting train into train and dev set

X = train.drop(columns=['label'])
Y = train.label

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.1)

#%% Normalizing features

X_train = X_train/255
X_dev = X_dev/255


#%% To one hot encoding

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoder.fit(Y_train.values.reshape(-1, 1))
Y_train = encoder.transform(Y_train.values.reshape(-1, 1)).toarray()
Y_dev = encoder.transform(Y_dev.values.reshape(-1, 1)).toarray()

#%% Visualizing the new data splitted

print("Formato de X_train: ", X_train.shape)
print("Formato de X_dev: ", X_dev.shape)
print("Formato de Y_train: ", Y_train.shape)
print("Formato de Y_dev: ", Y_dev.shape)

#%%
from keras.layers import Dense, Activation, BatchNormalization
from keras.models import Sequential
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import RMSprop, Adam


model = Sequential([
        Dense(units=32, input_shape=(784,), activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=10, activation='softmax')
        ])

#%% Model summary
model.summary()

#%% Compiling the model

model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])

#%% Training the model

model.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)

#%% Evaluating the model

model.evaluate(x=X_dev, y=Y_dev)

#%% Predicting on test set

predicted_one_hot_encoder = model.predict(test)
predicted_original_values = encoder.inverse_transform(predicted_one_hot_encoder)

#%% To csv

df = pd.DataFrame(predicted_original_values)
df.to_csv(r"/home/lucasgr/Área de trabalho/digit-recognizer/y_final.csv", index=True, index_label='ImageId', header=['Label'])


