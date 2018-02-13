from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

### dependencies for tensorflow spatials softmax
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

from utils import load_data

def spatial_softmax(features,
                    temperature=None,
                    name=None,
                    variables_collections=None,
                    trainable=True,
                    data_format='NHWC'):
  """Computes the spatial softmax of a convolutional feature map.
  First computes the softmax over the spatial extent of each channel of a
  convolutional feature map. Then computes the expected 2D position of the
  points of maximal activation for each channel, resulting in a set of
  feature keypoints [x1, y1, ... xN, yN] for all N channels.
  Read more here:
  "Learning visual feature spaces for robotic manipulation with
  deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.
  Args:
    features: A `Tensor` of size [batch_size, W, H, num_channels]; the
      convolutional feature map.
    temperature: Softmax temperature (optional). If None, a learnable
      temperature is created.
    name: A name for this operation (optional).
    variables_collections: Collections for the temperature variable.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
  Returns:
    feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
      the expected 2D locations of each channel's feature keypoint (normalized
      to the range (-1,1)). The inner dimension is arranged as
      [x1, y1, ... xN, yN].
  Raises:
    ValueError: If unexpected data_format specified.
    ValueError: If num_channels dimension is unspecified.
  """
  with variable_scope.variable_scope(name, 'spatial_softmax'):
    shape = array_ops.shape(features)
    static_shape = features.shape
    if data_format == 'NHWC':
      height, width, num_channels = shape[1], shape[2], static_shape[3]
    elif data_format == 'NCHW':
      num_channels, height, width = static_shape[1], shape[2], shape[3]
    else:
      raise ValueError('data_format has to be either NCHW or NHWC.')
    if num_channels.value is None:
      raise ValueError('The num_channels dimension of the inputs to '
                       '`spatial_softmax` should be defined. Found `None`.')

    with ops.name_scope('spatial_softmax_op', 'spatial_softmax_op', [features]):
      # Create tensors for x and y coordinate values, scaled to range [-1, 1].
      pos_x, pos_y = array_ops.meshgrid(
          math_ops.lin_space(-1., 1., num=height),
          math_ops.lin_space(-1., 1., num=width),
          indexing='ij')
      pos_x = array_ops.reshape(pos_x, [height * width])
      pos_y = array_ops.reshape(pos_y, [height * width])

      if temperature is None:
        temp_initializer = init_ops.ones_initializer()
      else:
        temp_initializer = init_ops.constant_initializer(temperature)

      if not trainable:
        temp_collections = None
      else:
        temp_collections = utils.get_variable_collections(
            variables_collections, 'temperature')

      temperature = variables.model_variable(
          'temperature',
          shape=(),
          dtype=dtypes.float32,
          initializer=temp_initializer,
          collections=temp_collections,
          trainable=trainable)
      if data_format == 'NCHW':
        features = array_ops.reshape(features, [-1, height * width])
      else:
        features = array_ops.reshape(
            array_ops.transpose(features, [0, 3, 1, 2]), [-1, height * width])

      softmax_attention = nn.softmax(features / temperature)
      expected_x = math_ops.reduce_sum(
          pos_x * softmax_attention, [1], keep_dims=True)
      expected_y = math_ops.reduce_sum(
          pos_y * softmax_attention, [1], keep_dims=True)
      expected_xy = array_ops.concat([expected_x, expected_y], 1)
      feature_keypoints = array_ops.reshape(expected_xy,
                                            [-1, num_channels.value * 2])
      feature_keypoints.set_shape([None, num_channels.value * 2])
  return feature_keypoints

def spatial_softmax_fun(x):
    return spatial_softmax(x)

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
model.add(Dense(num_classes, activation='softmax'))

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
#model.add(Dense(num_classes, activation='softmax'))

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




