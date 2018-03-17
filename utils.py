import zipfile
import os
import numpy as np

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

def load_data(filepath_data, filepath_labels, data_size):
    train_size = 0.85*data_size
    test_size = 0.15*data_size 
    
    labels_x,labels_y = np.loadtxt(filepath_labels, delimiter=',', usecols=(0, 1), unpack=True)
    labels_x_train = labels_x[0:train_size]
    labels_x_test = labels_x[train_size:train_size+test_size]
    labels_y_train = labels_y[0:train_size]
    labels_y_test = labels_y[train_size:train_size+test_size]
    
    datazip = zipfile.ZipFile(filepath_data)
    image_filenames = datazip.namelist()
    
    with datazip.open(image_filenames, 'r') as filename:
        data = np.frombuffer(datazip.read(filename), np.uint8, offset=16) #.reshape(len(data_size), image_size, image_size, channel_size)
        
    data_train = data[0:train_size,:,:,:]
    data_test = data[train_size:data_size,:,:,:]    
            
    return (data_train, labels_x_train, labels_y_train), (data_test, labels_x_test, labels_y_test)            

def load_batch(filepath_data, filepath_labels, data_size):
    train_size = 0.85*data_size
    test_size = 0.15*data_size 
    
    labels_x,labels_y = np.loadtxt(filepath_labels, delimiter=',', usecols=(0, 1), unpack=True)
    labels_x_train = labels_x[0:train_size]
    labels_x_test = labels_x[train_size:train_size+test_size]
    labels_y_train = labels_y[0:train_size]
    labels_y_test = labels_y[train_size:train_size+test_size]
    
    datazip = zipfile.ZipFile(filepath_data)
    image_filenames = datazip.namelist()
    
    with datazip.open(image_filenames, 'r') as filename:
        data = np.frombuffer(datazip.read(filename), np.uint8, offset=16) #.reshape(len(data_size), image_size, image_size, channel_size)
        
    data_train = data[0:train_size,:,:,:]
    data_test = data[train_size:data_size,:,:,:]    
            
    return (data_train, labels_x_train, labels_y_train), (data_test, labels_x_test, labels_y_test)            



#def load_batch(keys,labels,batch_size, img_dims, label_dims, clip_image=False, use_mean = False):    
#    nb = len(keys)                       
#    #count=0                                                                                                                                                                                                                                                                                                                                                                     
#    while True:
#        xdim = (batch_size,) + img_dims
#        X_train = np.zeros(xdim,dtype='float32')
#        Y_train = np.zeros((batch_size, label_dims),dtype='float32')
#        #count = count+1
#        for i in range(batch_size):
#            index = np.random.randint(nb)   
#            img = read_image(keys[index])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
#            
#            #print('Image shape pre-clip:', img.shape)
#            if clip_image is True:
#                img = img[:,37:(223-37),:]
#                #print('Image shape post-clip:', img.shape)        
#            img = img / 255.0
#            
##            if use_mean:
##                img = img - 0.5
#                
#            img = img.astype('float32')            
#            X_train[i] = img
#            
#            label = labels[index]
#            #label = label.reshape(1, label_dims)
#            Y_train[i] = label     
#            #print(key)
#            #print(label)
#        #print(count)
#        Y_train = Y_train.astype('float32')   
#        yield X_train, Y_train
        
#from keras.callbacks import Callback
#class History(Callback):
#    """Callback that records events into a `History` object.
#
#    This callback is automatically applied to
#    every Keras model. The `History` object
#    gets returned by the `fit` method of models.
#    """
#
#    def on_train_begin(self, logs=None):
#        self.epoch = []
#        self.history = {}
#
#    def on_epoch_end(self, epoch, logs=None):
#        logs = logs or {}
#        self.epoch.append(epoch)
#        for k, v in logs.items():
#            self.history.setdefault(k, []).append(v)        

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

#def generate_batches_from_hdf5_file(filepath, batchsize):
#    """
#    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
#    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
#    :param int batchsize: Size of the batches that should be generated.
#    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
#    """
#    dimensions = (batchsize, 28, 28, 1) # 28x28 pixel, one channel
# 
#    while 1:
#        f = h5py.File(filepath, "r")
#        filesize = len(f['y'])
#
#        # count how many entries we have read
#        n_entries = 0
#        # as long as we haven't read all entries from the file: keep reading
#        while n_entries < (filesize - batchsize):
#            # start the next batch at index 0
#            # create numpy arrays of input data (features)
#            xs = f['x'][n_entries : n_entries + batchsize]
#            xs = np.reshape(xs, dimensions).astype('float32')
#
#            # and label info. Contains more than one label in my case, e.g. is_dog, is_cat, fur_color,...
#            y_values = f['y'][n_entries:n_entries+batchsize]
#            ys = np.array(np.zeros((batchsize, 2))) # data with 2 different classes (e.g. dog or cat)
#
#            # Select the labels that we want to use, e.g. is dog/cat
#            for c, y_val in enumerate(y_values):
#                ys[c] = encode_targets(y_val, class_type='dog_vs_cat') # returns categorical labels [0,1], [1,0]
#
#            # we have read one more batch from this file
#            n_entries += batchsize
#            yield (xs, ys)
#        f.close()           