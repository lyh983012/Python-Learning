import fcnet as fc
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import image_loader as il


learning_rate=10

# padding（）：input矩阵周围补padding行的0
def padding(input_vector,
            padding):
    channel = input_vector.shape[0]
    input_height = input_vector.shape[1]
    input_width = input_vector.shape[2]
    input_padding = []
    print(input_height,input_width)

    for i in range(channel):
        temp = np.zeros((input_height + padding * 2, input_width + padding * 2))
        print(temp)
        for j in range(input_height-1):
            for k in range(input_width-1):
                print(i,j,k)
                temp[j + padding][k + padding] = input_vector[i][j][k]
        input_padding.append(temp)
    return input_padding

#用getpatch截取outputarray中第ij元素对应的一个高维矩阵中的一块
def get_patch(input_vector,
              i,
              j,
              kernel_width,
              kernel_height,
              stride):
    i = stride * i
    j = stride * j
    temp = input_vector[0:, i:i + kernel_height, j:j + kernel_width]
    return temp

#通过给定的output_array决定尺寸，适用给定的input_array和kernel_arra，输入bias，进行卷积操作，输出未加权的矩阵
def conv(input_array,
         kernel_array,
         output_array,
         stride,
         bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
         for j in range(output_width):
             output_array[i][j] = (
                                         get_patch(input_array, i, j, kernel_width,
                                                   kernel_height, stride) * kernel_array
                                     ).sum() + bias
    return output_array

#定义一下权重矩阵类
class filter:

    def __init__(self,
                 channel,
                 height,
                 width):
        self.weights=np.random.uniform(-1e-4,1e-4,(channel,height,width))
        self.bias=0
        self.grad=np.zeros(self.weights.shape)
        self.bias_grad=0

#定义ReLU函数
class activator:
    def forward(self,x):
        if x>0:
            return x
        else :
            return 0

    def backward(self,x):
        if x>0:
            return 1
        else:
            return 0

#定义卷积层，含有filter（权重矩阵组）完成正向传播、反向传播的接口
class convlayer:

    def __init__(self,addzero,
                 stride,
                 input_height,
                 input_width,
                 input_channel,
                 filter_height,
                 filter_width,
                 filter_number):

        self.input_height=input_height
        self.input_width=input_width
        self.input_channel=input_channel
        ''''''
        self.filter_height=filter_height
        self.filter_width=filter_width
        self.filter_number=filter_number
        ''''''
        self.addzero=addzero
        self.stride=stride
        self.output_height=int((self.input_height-
                                self.filter_height+2*addzero)/self.stride+1)#计算输出矩阵尺寸，取整操作很可能很危险
        self.output_width=int((self.input_width-
                               self.filter_width+2*addzero)/self.stride+1)#计算输出矩阵尺寸，取整操作很可能很危险
        self.outputs=np.zeros((self.filter_number,
                               self.output_height,
                               self.output_width))
        #输出的矩阵和卷积核的组数一样多
        ''''''
        self.filters=[]
        # 卷积核的层数和输入矩阵的层数一样多
        for i in range(self.input_channel):
            self.filters.append(filter(self.input_channel,
                                       self.filter_height,
                                       self.filter_width))
        self.activator = activator()
        self.learning_rate = learning_rate

    def forward(self,input_vector):
        temp=[]
        for i in range(len(self.filters)):
            temp.append(conv(input_vector,self.filters[i].weights, self.outputs, self.stride, self.filters[i].bias))
        return temp

