import math
import random
import matplotlib.pyplot as plt
import time

class PID_controller(object):
    def __init__(self,aim):
        self.I=0.0
        self.etlast=0.0
        self.et=0.0
        self.Kp=100#P系数,越大稳定越快，越容易失稳,越不易出现定值偏差
        self.Ki=0.2#I系数，越大稳定越快，系统静差消除时间变长,upwindows用于防发散
        self.Kd=0.00#d系数,微分环节主要作用是在响应过程中抑制偏差向任何方向的变化，will降低系统抗噪声能力
        self.timeC=1000#1000是个乱写的系数
        self.aim=aim
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.lastaim=self.aim
        self.upwindow=10

    def refresh(self):  #重新设aim之后的清零工作
        self.I=0.0
        self.etlast=0.0
        self.et=0.0
        self.sample_time = 0.00
        self.current_time = time.time()
        self.last_time = self.current_time
        self.lastaim=self.aim
        self.upwindow=self.aim*1.5

    def set_K(self,Ki=1.0,Kp=1.2,Kd=0.001):  #直接设置Kp，Ki,Kd
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd

    def PID(self,ui): #控制部分

        if self.aim != self.lastaim:
            self.refresh()
        self.current_time = time.time()
        dt=(self.current_time-self.last_time)*self.timeC
        self.et = self.aim-ui
        self.I += self.et * dt

        GI = self.Ki * self.I
        if (GI < -self.upwindow):
            GI = -self.upwindow
        elif (GI > self.upwindow):
            GI = self.upwindow
        GP = self.Kp * self.et
        if dt>0:
            GD = self.Kd * (self.et - self.etlast )/dt
        else :
            GD = 0

        uo = GP + GI + GD

        self.last_time = self.current_time
        self.etlast = self.et

        return uo

def G(x,i):
    return 100/(500*x+1)

if __name__ == '__main__':

    plt.axis([0, 2000, 0, 1000])
    ui=0
    controller=PID_controller(0)
    controller.aim = 10
    for i in range(1999):
        uo = controller.PID(ui)
        print("i=",i,"uo=",uo,"ui=",ui)
        plt.scatter(i, ui, c = 'r',marker = '.',s = 3)
        ui=G(uo,i)
        if (i==1000):
            controller.aim=50
            controller.refresh()
    plt.show()

'''
使用条件：
    已确定ui的目标值aim（也就是目标转速，ui最后将稳定到aim）
使用方法：
    1、初始化 controller=PID_controller(x)，x是uo目标值
    2、输入 controller.PID(ui) 在我们的车里ui应该是读到的转速
    3、读出 uo=controller.PID(ui)，uo作为给FPGA的参数
    4、当需要更改转速，使用controller.aim= newaim 即可
    
    可以设4个controller对4个轮子分别操控，每个周期只需要改转速，并且初始化controller
    left和right里的x相应地应该改成uo=controller.PID(ui)

controller1=PID_controller(0)  
def left(x):
    GPIO.output(13,0)
    if x>0:
        GPIO.output(2,0)
        controller1.aim=x
        parallel(controller1.PID(read(0)))#用对应转速来做PID控制
'''









