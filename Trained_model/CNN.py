


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import cv2
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam


data_train_t = np.load('data_train_correct.npy')
labels_train_t = np.load('labels_train_corrected.npy')

print(data_train_t.shape, labels_train_t.shape)


D=100
X = data_train_t.T
t = labels_train_t
X = np.array([ cv2.resize(x.reshape(300,300),(D,D)).reshape(D,D) for x in X ])
X_train, X_test, Y_train, Y_test = train_test_split(X, t, random_state=0, test_size = 0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(X_train)



model = Sequential()
# 1st Convolution Layer
model.add(Conv2D(input_shape=(100,100,1), filters=64, kernel_size=(3,3), padding='same', activation='relu'))
# 2nd Convolution layer
model.add(Conv2D(filters=64, kernel_size=(3,3),padding='same', activation='relu'))
# 3rd Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# 4th Convolution Layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# 5th Convolution layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
# 6th Pooling layer
model.add(MaxPooling2D(strides=(2,2), pool_size=(2,2)))
# 7th Convolution layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# 8th Convolution layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# 9th Convolution layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# 10th Pooling Layer
model.add(MaxPooling2D(strides=(2,2), pool_size=(2,2)))
# 11th Convolution layer
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='same', activation='relu'))
# 12th Conv layer
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='same', activation='relu'))
# 13th Conv layer
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='same', activation='relu'))
# 14th Pooling Layer 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# 15th Convolution layer
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='same', activation='relu'))
# 16th Conv layer
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='same', activation='relu'))
# 17th Conv layer
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='same', activation='relu'))
# 18th Pooling Layer 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(11, activation='softmax'))
model.summary()


opt = Adam(learning_rate=0.001)
model.compile( optimizer=opt, loss= tensorflow.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(X_train.reshape(-1,100,100,1), Y_train, epochs=100, batch_size = 8, validation_split= 0.2)

