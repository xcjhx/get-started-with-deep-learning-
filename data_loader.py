#引入pytorch的数据集库
from torch.utils.data import Dataset
#引入os，用于做文件路径控制
import os
#引入pandas，用于将数据以表格形式读取与处理
import pandas as pd
#引入numpy，用于处理数据
import numpy as np
#引入torch，用于以张量形式处理数据
import torch

#从Dataset为父类继承出iris_dataloader类
class iris_dataloader(Dataset):
    #重构对象初始化函数，有self与data_path两个参数
    #self:作为变量给类提供引用实例对象本身的方法，以修改对象属性值
    #data_path:作为实例变量--数据路径，传入对象
    def __init__(self, data_path):
        #让数据路径赋值到self以下
        self.data_path = data_path

        #判断地址是否存在，若不存在，返回“dataset dose not exits”
        assert os.path.exists(self.data_path),"dataset dose not exits"

        #使用pandas库读取数据，并将各列以“0 1 2 3 4”从左到右进行命名
        df = pd.read_csv(self.data_path, names=[0,1,2,3,4])

        #定义替换规则，冒号前的待替换的内容，冒号后的是替换后的内容
        d = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}

        #将df的第四列通过规则d进行替换赋值给df的第四列
        df[4] = df[4].map(d)

        #将数据集分为数据与标签两部分
        data = df.iloc[:,:4]
        label = df.iloc[:,4:]

        #做归一化处理，数据减去平均值/标准差
        data = (data - np.mean(data) / np.std(data))

        #先将数据转换为‘float32’类型的numpy数组，再将其转换为张量便于使用pytorch做数据处理
        self.data = torch.from_numpy(np.array(data, dtype='float32')) 
        self.label = torch.from_numpy(np.array(label, dtype='int64'))

        self.data = list(self.data)
        self.label = list(self.label)

        #求解并打印数据集大小
        self.data_num = len(label)
        print("当前数据集的大小：",self.data_num)

    def __len__(self):
        return self.data_num


    def __getitem__(self,index):
        
        return self.data[index], self.label[index]
