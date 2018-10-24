import random
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import math
import image_loader as il
from multiprocessing import Pool
import os, time

rate =10
beta=0.9
dropout=0.9
count = 0
layer_count = 3

#改进：需要使用dropout时随机地去掉一些边
'''
datas = (
    ((0.903, 0.903), (22.0 ,22.0)),
    ((0.953, 0.953), (42.0 ,42.0)),
    ((0.967, 0.967), (56.0,56.0)),
    ((0.973, 0.973), (66.0,66.0)),
    ((0.983, 0.983), (80.0,80.0)),
    ((0.989, 0.989), (94.0,94.0)),
    ((0.994, 0.994), (105.0,105.0)),
    ((0.996, 0.996), (107.0,107.0)),
    ((0.999, 0.999), (111.0,111.0)),
    ((1.001, 1.001), (111.0,111.0)),
    ((1.003, 1.003), (109.0,109.0)),
    ((1.006, 1.006), (106.0,106.0)),
    ((1.008, 1.008), (101.0,101.0)),
    ((1.011, 1.011), (95.0,95.0)),
    ((1.015, 1.015), (84.0,84.0)),
    ((1.023, 1.023), (68.0,68.0)),
    ((1.037, 1.037), (48.0,48.0)),
    ((1.079, 1.079), (24.0,24.0)))
temp1 = []
temp2 = []
temp = []
data2 = []
for i in range(len(datas)):
    temp1.append((datas[i][0][0] - 1) * 5)
    temp1.append((datas[i][0][1] - 1) * 5)
    temp2.append((datas[i][1][0]) / 180)
    temp2.append((datas[i][1][1]) / 180)
    temp.append(temp1)
    temp.append(temp2)
    data2.append(temp)
    temp1 = []
    temp2 = []
    temp = []
datas = data2
print(datas)
'''

datas=[]
temp1=il.read_image('t10k-images-idx3-ubyte')
temp2,n=il.read_label('t10k-labels-idx1-ubyte')
print(len(temp1),len(temp2))
for i in range(len(temp2)):
    temp = []
    temp.append(temp1[i])
    temp.append(temp2[i])
    datas.append(temp)
print(datas)

def f0(x):
    #return 1/(1+np.exp(-x))
    return np.tanh(x)
def df0(x):
    #return f0(x)*(1-f0(x))
    return 1-f0(x)*f0(x)
def caloutput(x,y):
    return x+y.weight*y.upnode.output
def caldelta(x,y):
    return x+y.weight*y.downnode.delta

class node(object):

    def __init__(self,node_i,layer_i):

        self.node_i = node_i
        self.layer_i = layer_i
        self.upstream = []  #储存上游的connection们
        self.downstream = []  #储存下游的connection们
        self.output = 0
        self.delta = 0
        self.store = 0
        self.delta0 = 0


    def add_upstream(self, conection):
        self.upstream.append(conection)
    def add_downstream(self, conection):
        self.downstream.append(conection)
    def output_cal(self):
        self.output=f0(reduce(caloutput, self.upstream,0)) #用于计算output，用caloutput直接遍历连接对象数组
        if ((random.uniform(0, 1) > dropout) and self.layer_i != layer_count - 1):
            node.output = 100000
    def inputlayer_output_cal(self,x):
        self.output = f0(x)  # 用于计算output，用caloutput直接遍历连接对象数组
        self.delta0 = df0(x)
    '''
        需要把x传进来，而output——cal是在layer中完成的，所以应当在layer传入x
    '''
    def delta_cal(self):
        temp=reduce(caldelta,self.downstream,0) #计算下一层的所有delta的加权求和
        self.delta=self.delta0*temp
    def outlayer_delta_cal(self,label): #计算隐藏层的delta，在node处分类好，在connection调用的时候就不会那么难受了
        self.store=(label - self.output)
        self.delta = self.delta0 * self.store
    '''
    需要把lable传进来，而delta——cal是在layer中完成的，所以应当给layer中传人lable
    '''
    def __str__(self):

        node_str = '%u-%u: output: %f delta: %f' % (self.layer_i, self.node_i, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
#神经元节点
class biasnode(object):
    def __init__(self,node_i,layer_i,x=0):
        self.node_i = node_i
        self.layer_i = layer_i
        self.upstream = []  #储存上游的connection们
        self.downstream = []  #储存下游的connection们
        self.output = 0
        self.delta = 0

    def output(self):
        return self.output

    def add_upstream(self,conection):
        self.upstream.append(conection)

    def add_downstream(self, conection):
        self.downstream.append(conection)

    def output_cal(self):
        self.output=0#用于计算output，用caloutput直接遍历连接对象数组

    def inputlayer_output_cal(self,x):
        self.output = 0 # 用于计算output，用caloutput直接遍历连接对象数组

    def delta_cal(self):
        self.delta= 0

    def outlayer_delta_cal(self,label): #计算隐藏层的delta，在node处分类好，在connection调用的时候就不会那么难受了
        self.delta = 0

    def __str__(self):

        node_str = '%u-%u: output: %f delta: %f' % (self.layer_i, self.node_i, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
#偏置项
class layer(object):
    def __init__(self,layer_i,node_count):
        self.nodes = []
        self.layer_i = layer_i
        self.node_count = node_count
        for node_i in range(self.node_count-1):
            self.nodes.append(node(node_i, self.layer_i))
        if (self.layer_i != layer_count - 1):
            self.nodes.append(biasnode(self.node_count-1, self.layer_i))
        else:
            self.nodes.append(node(self.node_count - 1, self.layer_i))
    def cal_layer_output(self,x=0):
        if ( self.layer_i == 0 ):
            for node in self.nodes:
                node.inputlayer_output_cal(x[node.node_i])
        else:
            for node in self.nodes:
                node.output_cal()
    def cal_layer_delta(self,lable=0):
        if(self.layer_i == layer_count-1):
            for node in self.nodes:
                node.outlayer_delta_cal(lable[node.node_i])  #lable，input都不作为成员，而是计算的时候作为参数传入
        else:
            for node in self.nodes:
                node.delta_cal()
#点集
class connection(object):

    def __init__(self, downnode,upnode):
        self.weight=random.uniform(-0.1,0.1)
        self.output=0
        self.gra=0
        self.v1=0
        self.oldgra=0
        self.upnode=upnode
        self.downnode=downnode
    def cal_weight(self):
        self.gra = self.downnode.delta * self.upnode.output
        self.v1=beta*self.oldgra+(1-beta)*self.gra
        self.weight=self.weight+rate*self.v1
        self.oldgra = self.gra

    def __str__(self):
        '''
        打印连接信息
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upnode.layer_i,
            self.upnode.node_i,
            self.downnode.layer_i,
            self.downnode.node_i,
            self.weight)
#点对点的连接
class connections(object):
    def __init__(self,layerup,layerdown):
        self.conns=[]
        for downnode in layerdown.nodes:
            for upnode in layerup.nodes:
                temp=connection(downnode,upnode)
                self.conns.append(temp)
                upnode.downstream.append(temp)
                downnode.upstream.append(temp)
    def change(self):
        for conn in self.conns:
            conn.cal_weight()
#层与层的连接
class net(object):

    def __init__(self, layer_count, nodelist):
        self.layers = []
        self.datasheet=[] #二维数组储存输入
        self.lablesheet=[] #二维数组储存标记
        self.all_connections = []
        for i in range(layer_count):
            self.layers.append(layer(i,nodelist[i]))
        for i in range(layer_count-1):
            self.all_connections.append(connections(self.layers[i],self.layers[i+1]))
        self.datas=[]
        self.data=[]
        self.lable=[]

    def print_information(self):
        print(len(mynet.layers))
        print(len(mynet.all_connections))

        for i in range(layer_count - 1):
            for connection in mynet.all_connections[i].conns:
                print(connection)

        for i in range(layer_count):
            for node in self.layers[i].nodes:
                print(node)

    def get_delta(self):
        num=0
        for node in self.layers[layer_count-1].nodes:
            num+=node.store*node.store
        num=math.sqrt(num/len(self.layers[layer_count-1].nodes))
        return num
    def get_data(self,datas):
        self.datas=datas

    def cal_netoutput(self):
        for layer in self.layers:
                layer.cal_layer_output(self.data)
    def cal_netdelta(self):
        for index in range(len(self.layers),0,-1):
            layer=self.layers[index-1]
            layer.cal_layer_delta(self.lable)
        for index2 in range(len(self.all_connections),0,-1):
            conns=self.all_connections[index2-1]
            conns.change()

    def train_net_once(self,index=0):
        self.data=self.datas[index][0]
        self.lable=self.datas[index][1]
        self.cal_netoutput()
        self.cal_netdelta()

    def predict(self,data):
        for layer in self.layers:
            layer.cal_layer_output(data)
        return self.layers[layer_count - 1].nodes[0].output
    def print(self, data):
        for layer in self.layers:
            layer.cal_layer_output(data)
        for node in self.layers[layer_count - 1].nodes:
            print(node.output)
        #####temp
        return self.layers[layer_count - 1].nodes[0].output




if __name__ == '__main__':


    plt.axis([0, 1000, 0, 0.00001])
    #########数据归一化
    mynet = net(layer_count, (len(datas[0][0]), 25, len(datas[0][1])))
    count1=0
    mynet.get_data(datas)
    print(mynet)
    plt.show()
    time_start = time.time()
    while (1):
        mynet.train_net_once(count1)
        count1 += 1
        plt.scatter(count1, mynet.get_delta())
        if(abs(mynet.get_delta()<0.000001)):
             time_end = time.time()
             print('totally cost', time_end - time_start)
             break
    plt.clf()
    plt.show()
    x=np.linspace(900,1100,100)
    y = (mynet.predict(((x / 1000 - 1)*5,(x / 1000 - 1)*5)) * 180)
    plt.plot(x, y,'.')
    plt.show()


