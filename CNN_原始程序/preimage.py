# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:10:03 2017

@author
"""

import numpy as np
import pickle
#import tensorflow as tf
#from unittest.mock import MagicMock
import random
import time



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
    t = a + ( (x - grayscale_min)*(b - a))/( grayscale_max - grayscale_min )
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
处理并保存所有数据，train：95%    test：5%
'''
def load_image(batch_id):

    with open(r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\data_line\data_line_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file)#, encoding='latin1'

    features = np.array(batch['data']).reshape((len(batch['data']), 3, 224, 224)).transpose(0, 2, 3, 1)
    #features = np.array(batch['data']).reshape((len(batch['data']), 3, 112, 112)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    
    print(len(features))
    
    #打乱图片顺序 (若修改，将_preprocess_and_save也要修改)
    ind = [i for i in range(len(features))]
    random.shuffle(ind)
    features = features[ind]
    labels = one_hot_encode(labels)
    labels = labels[ind]
    print(len(labels))
    
    return features, labels


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    #labels = one_hot_encode(labels)
    labels = labels
    #print (labels)
    pickle.dump((features, labels), open(filename, 'wb'))
    
n_batches = 16
valid_features = []
valid_labels = []
start = time.time()
for batch_i in range(1,n_batches+1):
    #print(batch_i)
    features,labels = load_image(batch_i)
    print(np.shape(features))

    validation_count = int(len(features) * 0.05) #测试集长度
    
    _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\p\preprocess_batch_' + str(batch_i) + '.p')  # 生成训练集
            #'../Stock/line/data_batch/line_p/2/preprocess_batch_' + str(batch_i) + '.p')  # 生成训练集
     
    valid_features.extend(features[-validation_count:])
    valid_labels.extend(labels[-validation_count:])
    
_preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\p\preprocess_validation.p')   #生成测试集
        #'../Stock/line/data_batch/line_p/2/preprocess_validation.p')  # 生成训练集

end = time.time()
time_cha_value = end - start
print ("用时:" + str(time_cha_value) + '秒')
