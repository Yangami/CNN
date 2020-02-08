# encoding: utf-8
# 第一个检查点,预处理的数据已保存到本地。
# DON'T MODIFY ANYTHING IN THIS CELL
from __future__ import print_function
import pickle
import tensorflow as tf
import numpy as np
import time
# Load the Preprocessed Validation data  preprocess_validation.p
valid_features, valid_labels = pickle.load(open(r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\p\preprocess_validation.p', mode='rb'))
#valid_features, valid_labels = pickle.load(open('../Stock/line/data_batch/0608/p/preprocess_validation.p', mode='rb'))
print(np.shape(valid_features))


# 构建网络


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    image_input = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name='x')
    return image_input


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    label_input = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    return label_input


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return keep_prob


# 卷积和最大池化层
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    w = tf.Variable(
        tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs],
                            stddev=0.05))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs], stddev=0.05))
    x = tf.nn.conv2d(x_tensor, w, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1],
                       padding="SAME")
    return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# tests.test_con_pool(conv2d_maxpool)

# 扁平化层

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    x_tensor = tf.reshape(x_tensor, shape=[-1, x_shape[1] * x_shape[2] * x_shape[3]])

    return x_tensor
    # return None


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# tests.test_flatten(flatten)

# 全连接层

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    batch, size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size, num_outputs], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs], stddev=0.05))
    fc1 = tf.matmul(x_tensor, w)
    fc1 = tf.add(fc1, b)
    fc1 = tf.nn.relu(fc1)
    return fc1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# tests.test_fully_conn(fully_conn)

# 输出层

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    batch, size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size, num_outputs], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs], stddev=0.05))
    fc1 = tf.add(tf.matmul(x_tensor, w), b)
    return fc1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# tests.test_output(output)

# 构建网络
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv_num_outputs:滤波器的种类个数
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    con = conv2d_maxpool(x, 32, (5, 5), (2, 2), (2, 2), (2, 2))
    con = conv2d_maxpool(con, 64, (3, 3), (1, 1), (2, 2), (2, 2))
    con = conv2d_maxpool(con, 128, (2, 2), (1, 1), (2, 2), (2, 2))

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    con = flatten(con)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc = fully_conn(con, 512)  # 1024
    fc = tf.nn.dropout(fc, keep_prob)
    fc = fully_conn(fc, 128)  # 512
    fc = tf.nn.dropout(fc, keep_prob)
    fc = fully_conn(fc, 32)
    fc = tf.nn.dropout(fc, keep_prob)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(fc, 6)

    # TODO: return output
    return out


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((224, 224, 3))
#x = neural_net_image_input((112, 112, 3))
y = neural_net_label_input(6)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

#学习率自适应
learning_rate_start = 0.1  #初始学习率
decay_rate = 0.96    #衰减系数
global_steps = 300   #迭代次数
decay_steps = 100   #衰减速度

#global_ = tf.Variable(tf.constant(0))
global_ = tf.Variable(0)
learning_rate = tf.train.exponential_decay(learning_rate_start, global_, decay_steps, decay_rate, staircase=True)

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
# optimizer = tf.train.AdamOptimizer().minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step = global_)
# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))  # tf.argmax获取logits中的最大值的索引
# correct_pred = tf.equal(logits,y)
# jin = tf.cast(correct_pred, tf.float32)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')  # tf.cast将true变成float(1/0)


# tests.test_conv_net(conv_net)

# 训练神经网络
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function

    # session.run(tf.global_variables_initializer())
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})
    # pass


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""


# tests.test_train_nn(train_neural_network)

# 显示数据
def print_stats(session, feature_batch, label_batch, cost, learning_rate,accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    Loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    Accuracy = session.run(accuracy, feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    learnrate = session.run(learning_rate, feed_dict={x: batch_features, y: batch_labels})
    print('Loss  {:.6f} - Learning_rate {:.6f} - Accuracy {:.6f}'.format(Loss,learnrate,Accuracy))

# pass
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    #filename = '../Stock/line/data_batch/0608/p/preprocess_batch_' + str(batch_id) + '.p'  # 加载训练集
    filename = r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\p\preprocess_batch_' + str(batch_id) + '.p'  # 加载训练集
    # features, labels = pickle.load(open(filename, mode='rb'))
    features, labels = pickle.load(open(filename, 'rb'))

    # print(features.size)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


# 超参数
# TODO: Tune Parameters
epochs = 300  # 训练周期数
batch_size = 224
keep_probability = 0.75

# 训练模型
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
#save_model_path = './model/model_4/image_classification_4'
save_model_path = r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\image_classification_16'

print('Training...')
with tf.Session() as sess:
    start = time.time()
    # Initializing the variables
    # sess.run(tf.global_variables_initializer())
    tf.initialize_all_variables().run()

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 16
        for batch_i in range(1, n_batches+1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i),end='')

            #print(sess.run(learning_rate, feed_dict={x: batch_features, y: batch_labels}))

            print_stats(sess, batch_features, batch_labels, cost, learning_rate, accuracy)
            #print_learning_rate(sess,learning_rate)
    # plt.plot(epoch,accuracy)
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
    end = time.time()
    time_cha_value = end - start
    print("用时:" + str(time_cha_value) + '秒')