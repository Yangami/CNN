# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:27:38 2018

@author: admin
"""

import numpy as np
import pickle
import time
#import tensorflow as tf
#from unittest.mock import MagicMock




'''
归一化输入
'''
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (128, 128, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    a = 0
    b = 1
    grayscale_min = np.min(x)
    grayscale_max = np.max(x)
    t = a + ( (x - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min )
    return t 
'''
对标签编码
'''
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    x = np.array(x) 
    #print (x)
    num_labels = x.shape[0]
    x_one_hot = np.zeros((num_labels,6))
    for i in np.arange(num_labels):
        x_one_hot[i][x[i]] = 1
    return x_one_hot
       
'''
处理并保存所有数据
'''
def load_image(batch_id):

    with open(r'C:\Users\yy\Desktop\Stock\line\line\data_line_test_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file) #, encoding='latin1'

    features = np.array(batch['data']).reshape((len(batch['data']), 3, 224, 224)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    
    return features, labels
    
    
def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)
    print (np.shape(features),np.shape(labels))
    pickle.dump((features, labels), open(filename, 'wb'))

    
n_batches = 1

start = time.time()
for batch_i in range(1,n_batches+1):
    #print(batch_i)
    features,labels = load_image(batch_i)
    _preprocess_and_save(
            normalize,
            one_hot_encode,
            features,
            labels,          
            r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\p\preprocess_shibie_' + str(batch_i) + '.p')
end = time.time()
time_cha_value = end - start
print ("用时:" + str(time_cha_value) + '秒')
