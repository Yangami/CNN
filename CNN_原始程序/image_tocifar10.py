# encoding: utf-8
import pickle
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg
import time

class DictSave(object):
    def __init__(self,filenames,classnum):
        self.filenames = filenames
        self.classnum = classnum

    def image_input(self,filenames,classnum):
        i = 0
        self.all_arr = [[0 for m in range(6000)] for n in range(len(filenames))]  #700指特征的个数
        self.label_all_arr = [[0] for k in range(len(filenames))]
        for filename in filenames:
            j = 0
            self.arr, self.label_arr = self.read_file(filename,classnum)
            self.all_arr[i][j:len(self.arr)] = self.arr
            self.label_all_arr[i][j:len(self.label_arr)] = self.label_arr
            i = i+1

    def read_file(self,filename,classnum):
        #im = Image.open(os.path.join("../Stock/up/up_train/up_%s"%classnum,filename),'r')#打开一个图像
        im = Image.open(os.path.join(r"C:\Users\yy\Desktop\Stock\line\line\line_train\line_%s" % classnum, filename), 'r')  # 打开一个图像
        site=filename.find('.')
        label_arr = np.array([int(filename[site-1:site])])
       
       
        # 将图像的RGB分离
        r,g,b = im.split()
        # 将PILLOW图像转成数组
        r_arr = plimg.pil_to_array(r)
        g_arr = plimg.pil_to_array(g)
        b_arr = plimg.pil_to_array(b)

        # 将224*224二维数组转成50176的一维数组
        r_arr1 = r_arr.reshape(224*224)
        g_arr1 = g_arr.reshape(224*224)
        b_arr1 = b_arr.reshape(224*224)
        # 3个一维数组合并成一个一维数组,大小为150528
        arr = np.concatenate((r_arr1, g_arr1, b_arr1))
        
        return arr,label_arr
    def pickle_save(self,arr,label_arr,classnum):
        print ("正在存储")
       
        # 构造字典,所有的图像数据都在arr数组里,这里只存图像数据,没有存label
        data_batch_1 = {'data': arr ,'labels': label_arr}

        f = open(r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\data_line\data_line_%s' %classnum, 'wb')

        pickle.dump(data_batch_1, f)#把字典存到文本中去
        f.close()
        print ("第"+str(classnum)+"个文件夹存储完毕")
if __name__ == "__main__":
    start = time.time()
    N = 16    #压缩文件夹个数
    for classnum in range(1,N+1):

        path = r"C:\Users\yy\Desktop\Stock\line\line\line_train\line_%s" %classnum

        filenames = os.listdir(path)
        #print filenames
        ds = DictSave(filenames,classnum)
        ds.image_input(ds.filenames,ds.classnum)
        ds.pickle_save(ds.all_arr,ds.label_all_arr,ds.classnum)
        print ("line_%s"%classnum +"文件夹,最终数组的大小:"+str(np.shape(ds.label_all_arr))+str(np.shape(ds.all_arr)))
    end = time.time()
    time_cha_value = end - start
    print ("用时:"+ str(time_cha_value) + '秒')
