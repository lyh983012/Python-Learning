#coding=utf-8

#!/usr/bin/python
from functools import reduce

import numpy as np
print (np.version.version)

def f1(x,y):
    return x+y

#python中的高维列表实质上可以使用列表嵌套列表来实现,可以当成向量来使用
class perception(object):

    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0 for _ in range(input_num)]
        self.bias = 0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(reduce(f1,  list(map(lambda a_b: a_b[0]*a_b[1], zip(input_vec, self.weights))), 0.0) + self.bias) #其实可以不用reduce
        #return (input_vec[0]*self.weights[0]+ input_vec[1]*self.weights[1]+self.bias)

    def train(self, input_vecs, labels, iteration, rate):

        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):

        samples = zip(input_vecs, labels)

        for (input_vec, label) in samples:#这里括号中的应该是新的局部变量？
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = list(map(lambda x_w: x_w[1]+rate*delta*x_w[0],zip(input_vec, self.weights)))
        # 更新bias
        self.bias += rate * delta


def f(x):
    if x>0:
        return 1
    else :
        return 0

def get_training_datasheet():
    input_vec=[[1,1],[1,0],[0,1],[0,0]]
    lable=[1,1,1,0]
    return input_vec,lable

def taining_perception():
    p1=perception(2,f)
    input_vecs, labels=get_training_datasheet()
    p1.train(input_vecs, labels,100,0.10)
    return p1

if __name__ == '__main__':
    # 训练and感知器
    or_perception = taining_perception()
    # 打印训练获得的权重
    print (or_perception)
    # 测试
    print ('1 and 1 = %d' % or_perception.predict([1, 1]))
    print ('0 and 0 = %d' % or_perception.predict([0, 0]))
    print ('1 and 0 = %d' % or_perception.predict([1, 0]))
    print ('0 and 1 = %d' % or_perception.predict([0, 1]))
    while(1):
        i=input()
        print(i)

