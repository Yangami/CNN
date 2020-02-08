#encoding:utf-8
from __future__ import print_function
import pickle
import tensorflow as tf
import numpy as np
import os
import time


features,labels = pickle.load(open(r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\data_line\preprocess_shibie_1.p', mode='rb'))
#print (np.shape(features))
#只需改动此处与model有关数值即可
#--------------------------开始--------------------------
save_model_path = r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\model\image_classification_7'
test_dict = {0:'转折',1:'上升',2:'水平',4:'上升负样本',3:'转折负样本',5:'水平负样本'}
model_xushu = 7
#--------------------------终止--------------------------
#卷积和最大池化层
def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """ 
    # TODO: Implement Function
    image_input = tf.placeholder(dtype=tf.float32,shape=[None,image_shape[0],image_shape[1],image_shape[2]],name='x')
    return image_input


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    label_input = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='y')
    return label_input


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    return keep_prob


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_nn_image_inputs(neural_net_image_input)
#tests.test_nn_label_inputs(neural_net_label_input)
#tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

#卷积和最大池化层
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
    w = tf.Variable(tf.truncated_normal([conv_ksize[0],conv_ksize[1],x_tensor.get_shape().as_list()[3],conv_num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs],stddev=0.05))
    x=tf.nn.conv2d(x_tensor,w,[1,conv_strides[0],conv_strides[1],1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    x=tf.nn.relu(x)
    x=tf.nn.max_pool(x,[1,pool_ksize[0],pool_ksize[1],1],[1,pool_strides[0],pool_strides[1],1],padding="SAME")
    return x


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_con_pool(conv2d_maxpool)

#扁平化层

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    x_tensor = tf.reshape(x_tensor,shape=[-1,x_shape[1]*x_shape[2]*x_shape[3]])

    return x_tensor
    #return None


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_flatten(flatten)

#全连接层

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.05))
    fc1 = tf.matmul(x_tensor,w)
    fc1 = tf.add(fc1,b)
    fc1 = tf.nn.relu(fc1)
    return fc1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_fully_conn(fully_conn)

#输出层

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.05))
    fc1 = tf.add(tf.matmul(x_tensor,w),b)
    return fc1


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
#tests.test_output(output)
tf.reset_default_graph()
#构建网络
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
    fc = fully_conn(con, 512) #1024
    fc = tf.nn.dropout(fc, keep_prob)
    fc = fully_conn(fc, 128) #512
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

x = neural_net_image_input((224, 224, 3))
y = neural_net_label_input(6)
keep_prob = neural_net_keep_prob_input()

logits = conv_net(x, keep_prob)   #预测（识别）值
logits = tf.identity(logits, name='logits')
'''-------------------------方法一:将股票代码与时间一起显示-------------------------------
def read_filename(filenames):
    stockname = []
    for filename in filenames:
        site=filename.find('.')
        stockname.append(filename[0:site-2])

    return stockname
'''

'''-------------------------方法二:将股票代码与时间分开显示-------------------------------'''
#预测结果分类
def read_filename(filenames):
    stockname = []
    stocktimestart = []
    stocktimeend = []
    for filename in filenames:
        site=filename.find('.')
        stockname.append(filename[0:8])      #获取测试集中的股票代码,list
        stocktimestart.append(filename[9:19])   #获取测试集中预测股票的起始时间,list
        stocktimeend.append(filename[20:site-2])   #获取测试集中预测股票的结束时间,list

    return stockname,stocktimestart,stocktimeend

#结果预测保存
def text_save(content, filename, mode='a'):
    #mode = 'a'若filename无该文件,则会创建一个该文件,且原文件内容不覆盖
    #mode = 'w'则原文件内容会覆盖,最后仅剩一条数据
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]))
    file.write("\n")
    file.close()

saver = tf.train.Saver()
with tf.Session() as sess:
    start = time.time()
    saver.restore(sess,save_model_path)  
   
    print("Model restored.")
    pred_value = sess.run(logits,feed_dict={x:features,y:labels,keep_prob:1})
    #print (labels)
    #print(pred_value)
    labels_l = np.argmax(labels,1)
    pred_class_index=np.argmax(pred_value,1)
    #print (pred_class_index)

    path = r"C:\Users\yy\Desktop\Stock\line\line\line_test"
    filenames = os.listdir(path)
    # print (filenames)

    Stock_namelist,StocktimeStart,StocktimeEnd = read_filename(filenames)    #预测股票名称列表

    j,m= 0,0
    uplist = []
    linelist = []
    changelist=[]
    file_pro_path="../Stock/line/result_change_file_pro.txt"
    #print (labels_l)
    #print (pred_class_index)
    file = open(file_pro_path,'a')  #若改成'w'则覆盖,每次只显示一次运行结果
    file.write('-'*20+"在时间为:" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "model" + str(model_xushu) + "运行的结果"+'-'*20+ '\n')
    file.close()
    count = 0
    for i in pred_class_index:
        #print(str(i),labels_l[m])
        if str(i) == str(labels_l[m]):
            file_pro = (str(m+1) +"代码为", Stock_namelist[j], "从", StocktimeStart[j], "至", StocktimeEnd[j],
                        "股票图片预测:" + test_dict[i], "(" + str(i) + ")")
        else :
            file_pro = (str(m+1) + "代码为", Stock_namelist[j], "从", StocktimeStart[j], "至", StocktimeEnd[j],
                        "股票图片预测:" + test_dict[i], "(" + str(i) + ")", "原始值为:", labels_l[m])
            count += 1
        m += 1
        text_save(file_pro, file_pro_path)
    file = open(file_pro_path, 'a')  # 若改成'w'则覆盖,每次只显示一次运行结果
    file.write("model"+str(model_xushu)+"模型运行的结果识别错误个数共计: " + str(count) + "个"+'+'*20+ '\n')
    file.write("\n")
    file.close()
    '''
        if i == 1:
            uplist.append(Stock_namelist[j])
        elif i == 2:
            linelist.append(Stock_namelist[j])
        else:
            changelist.append(Stock_namelist[j])
        j += 1
        m += 1
    print("上升趋势股票代码", uplist)
    print("水平趋势股票代码", linelist)
    print("转折趋势股票代码", changelist)
    '''
    print ("识别错误个数为:%d"%count)
    end = time.time()
    time_cha_value = end - start
    print("用时:" + str(time_cha_value) + '秒')