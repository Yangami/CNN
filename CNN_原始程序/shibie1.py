# encoding: utf-8
import pickle
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg
import time

class DictSave(object):
    def __init__(self,filenames):
        self.filenames = filenames

    def image_input(self,filenames):
        i = 0
        self.all_arr = [[0 for m in range(6000)] for n in range(len(filenames))]
        self.label_all_arr = [[0] for k in range(len(filenames))]
        for filename in filenames:
            j = 0
            self.arr, self.label_arr = self.read_file(filename)
            self.all_arr[i][j:len(self.arr)] = self.arr
            self.label_all_arr[i][j:len(self.label_arr)] = self.label_arr
            i = i+1

    def read_file(self,filename):
        im = Image.open(os.path.join(r"C:\Users\yy\Desktop\Stock\line\line\line_test",filename),'r')#打开一个图像

        site=filename.find('.')
        label_arr = np.array([int(filename[site-1:site])])
       
        # 将图像的RGB分离
        r, g, b = im.split()
        # 将PILLOW图像转成数组
        r_arr = plimg.pil_to_array(r)
        g_arr = plimg.pil_to_array(g)
        b_arr = plimg.pil_to_array(b)

        # 将200*240二维数组转成48000的一维数组
        r_arr1 = r_arr.reshape(224*224)
        g_arr1 = g_arr.reshape(224*224)
        b_arr1 = b_arr.reshape(224*224)
        # 3个一维数组合并成一个一维数组,大小为144000
        arr = np.concatenate((r_arr1, g_arr1, b_arr1))
        
        return arr,label_arr
    def pickle_save(self,arr,label_arr):
        print ("正在存储")
       
        # 构造字典,所有的图像数据都在arr数组里,这里只存图像数据,没有存label
        data_up_test_1 = {'data': arr ,'labels': label_arr}
        
        f = open(r'C:\Users\yy\Desktop\神经网络\CNN\CNN_原始程序\data_line\data_line_test_1', 'wb')

        pickle.dump(data_up_test_1, f)#把字典存到文本中去
        f.close()
        print ("存储完毕")
if __name__ == "__main__":
    start = time.time()
    path = r"C:\Users\yy\Desktop\Stock\line\line\line_test"

    filenames = os.listdir(path)
    #print(filenames)
    ds = DictSave(filenames)
    ds.image_input(ds.filenames)
    ds.pickle_save(ds.all_arr,ds.label_all_arr)
    #print(ds.label_all_arr)
    print ("最终数组的大小:"+str(np.shape(ds.label_all_arr))+str(np.shape(ds.all_arr)))
    end = time.time()
    time_cha_value = end - start
    print ("用时:" + str(time_cha_value) + '秒')