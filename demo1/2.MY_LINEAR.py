
class linear_re(object):

    def __init__(self,stimulator,dimension):
        self.weight=[0 for _ in range(dimension) ]
        self.fun=stimulator
        self.X,self.Y=self.getdata()
        self.dimension=dimension

    def __str__(self):
        return "this is my own demo"

    def predict(self,Xi):
        _y=0
        for i in range(self.dimension):
            _y+=Xi[i]*self.weight[i]
        return self.fun(_y)

    def renew(self):
        for i in range(len(self.X)):
            temp1=self.delta * (self.Y[i] - self.predict(self.X[i]))
            for j in range(self.dimension):
                self.weight[j]=self.weight[j]+temp1*self.X[i][j]  #注意，是加号，是往梯度的反向！

    def train(self,n,delta):
        self.delta=delta
        for i in range(n):
            self.renew()

    def getdata(self):
        data_x=[[1,2,3,4],[1,3,5,4],[1,3,2,1],[1,3,2,6],[1,3,2,3]]
        lable_y=[1,2,3,2.5,4]
        return data_x,lable_y

'''
线性单元类设计的核心思路：构造函数用于初始化激活函数和维数
train函数用于传入循环次数和调整精度
采用随机梯度下降，每次值计算一个样本就进行更新（predict-renew）
因为类的激活函数很简单所以对xi求偏导后系数刚好就是weight（i）
更改激活函数后应当修改类内的renew上的系数
'''

def stimulator(x):     #感知器的函数，单层的可以拟合线性函数
    return x

if __name__ == '__main__':

    demo =linear_re(stimulator,4) #在此输入用来激活的函数和数据维数+1(第一位是常数)
    demo.train(100000,0.0001)    #输入训练次数和delta
    print('f(x1,x2,x3,...,xn)=',end=" ")
    print('%f +'% demo.weight[0],end=" ")
    for i in range(len(demo.weight)-1):
        print ('%f*x%d +' % ((demo.weight[i+1],i+1) ),end=" " )

    print("0")
    print (demo.predict([1,1,1,1]))
    print (demo.predict([1,2,2,2]))
    print (demo.predict([1,3,3,3]))







