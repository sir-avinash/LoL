import zipfile
import os
import numpy as np

def load_data():
    filepath_labels = 'lol_labels.txt'
    filepath_data = 'lol_images.zip'
    image_size = 224
    channel_size = 1
    
    data_size = 60000
    train_size = 0.85*data_size
    test_size = 0.15*data_size 
    
    labels_x,labels_y = np.loadtxt(filepath_labels, delimiter=',', usecols=(0, 1), unpack=True)
    labels_x_train = labels_x[0:train_size]
    labels_x_test = labels_x[train_size:train_size+test_size]
    labels_y_train = labels_y[0:train_size]
    labels_y_test = labels_y[train_size:train_size+test_size]
    
    datazip = zipfile.ZipFile(filepath_data)
    image_filenames = datazip.namelist()
#    data = np.zeros((data_size,image_size,image_size,channel_size)) #NHWC format
    
#    for iter, filename in enumerate(image_filenames):
#        try:
#            data[iter,:,:,:] = datazip.read(filename)
#        except KeyError:
#            print('ERROR: Did not find %s in zip file' %filename)
#        else:
#        print(filename, ':')
    
    with datazip.open(image_filenames, 'r') as filename:
        data_train = np.frombuffer(datazip.read(), np.uint8, offset=16).reshape(len(data_size), image_size, image_size, channel_size)
            
    return (data_train, labels_x_train, labels_y_train), (data_test, labels_x_test, labels_y_test)            
           