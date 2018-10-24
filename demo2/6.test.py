import fcnet as fc
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf


learning_rate=10

# padding（）：input矩阵周围补padding行的0
def padding(input_vector,
            padding):
    channel = input_vector.shape[0]
    input_height = input_vector.shape[1]
    input_width = input_vector.shape[2]
    input_padding = []

    for i in range(channel):
        temp = np.zeros((input_height + padding * 2, input_width + padding * 2))
        for j in range(input_height-1):
            for k in range(input_width-1):
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

def op_by_element(input_vctor,
                  op):
    for x in np.nditer(input_vctor,op_flags=['readwrite']):
        x[...]=op(x)



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

    def update(self):
        self.weights-=self.grad*learning_rate
        self.bias-=self.bias_grad *learning_rate

    def get_weight(self):
        return  self.weights

    def get_bias(self):
        return self.bias

#定义ReLU函数
class activator:

    def forward(self,x):
        if x>0:
            return x
        else:
            return 0

    def backward(self,x):
        if x>0:
            return 1
        else:
            return 0

#定义卷积层，含有filter（权重矩阵组）完成正向传播、反向传播的接口、含有自己的sensitity map
class convlayer:

    def __init__(self,addzero,
                 stride,
                 input_height,
                 input_width,
                 input_channel,
                 filter_height,
                 filter_width,
                 filter_number):

        self.input_array = 0
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


    # 通过给定的output_array决定尺寸，适用给定的input_array和kernel_arra，输入bias，进行卷积操作，输出未加权的矩阵
    def conv(self,
             input_array,
             kernel_array,
             output_array,
             stride,
             bias):

        channel_number = input_array.ndim #?
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

    def forward(self,input_vector):
        self.input_array=input_vector
        self.paadding_input_array=padding(input_vector,self.addzero)
        temp=[]
        for i in range(len(self.filters)):
            self.conv(self.input_array,self.filters[i].weights, self.outputs[i], self.stride, self.filters[i].bias)
        print('before',self.outputs)
        op_by_element(self.outputs, self.activator.forward)
        print('after', self.outputs)
        return temp

    def bp_sensitivity_map(self, sensitivity_array,
                           activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width +
              self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(map(
                lambda i: np.rot90(i, 2),
                filter.get_weights()))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                self.conv(padded_array[f], flipped_weights[d],
                     delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        op_by_element(derivative_array,
                        activator.backward)
        self.delta_array *= derivative_array

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height,
                                 expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = \
                    sensitivity_array[:, i, j]
        return expand_array







sess=tf.InteractiveSession()



input_array_test=np.array([[[-1,2,-1,1,1],[1,1,1,-10,1],[1,1,1,2,1],[1,1,1,2,1],[1,1,1,5,1]],[[1,1,1,2,1],[1,21,1,1,1],[1,1,1,3,1],[1,1,1,2,1],[1,1,1,5,1]]])
conv_layer1=convlayer(1,2,5,5,2,2,2,2)

conv_layer1.forward(input_array_test)
