from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from utils import load_data, spatial_softmax_fun

batch_size = 128
epochs = 10

img_size = 224      # input image dimensions
channel_size = 1
label_size = 1      # label dimensions

img_dims = (img_size, img_size, channel_size)
label_dims = (label_size, label_size)

filepath_labels = 'lol_labels.txt'
filepath_data = 'lol_images.zip'    
data_size = 60000
    
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape(x_train.shape[0], img_size, img_size, channel_size)
x_test = x_test.reshape(x_test.shape[0], img_size, img_size, channel_size)
input_shape = (img_size, img_size, channel_size)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


###### Model 0 ##############################
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(label_size, activation='softmax'))

###### Model 1 ##############################
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Lambda(spatial_softmax_fun))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(label_size, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




