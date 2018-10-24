import matplotlib.pyplot as mpb
import numpy as np


nc = [1, 1.5582, 1.71037, 1]   #结构数据，几个折射率，球面半径，厚度
nf = [1, 1.57596, 1.73468, 1]
nd = [1, 1.5688, 1.7172, 1]
r = [18.016, -9.819, -27.698]
d = [3, 1.2 , 0]
L = -28.33  #物距、最大孔径角、最大高度（第一个透镜的孔径）
Umax = 0.15
hmax = 4.25
du = Umax / 500

def refraction(U , L, r, n, n_):    #光线追迹
    I = np.arcsin((L-r)*np.sin(U)/r)
    I_ = np.arcsin(n*np.sin(I)/n_)
    U_ = U+I-I_
    if U_==0:
        U_=0.001
    L_ = r+r*np.sin(I_)/np.sin(U_)
    return U_ , L_

def transfer(U,L,d):    #  转面公式
    U=U
    d=d-abs(U)*20
    L=L - d
    return U,L

def imagine(U,L,ch):     #  对不同的U 迭代成像
    if ch == 1 :
        for i in range(2):
            U, L = refraction(U, L, r[i], nc[i], nc[i + 1])
            U, L = transfer(U, L, d[i])
    if ch == 2:
        for i in range(2):
            U, L = refraction(U, L, r[i], nf[i], nf[i + 1])
            U, L = transfer(U, L, d[i])
    if ch == 3 :
        for i in range(2):
            U, L = refraction(U, L, r[i], nd[i], nd[i + 1])
            U, L = transfer(U, L, d[i])

    return U,L


if __name__ == '__main__':

    dy1=[]
    dy2=[]
    dy3=[]
    z=[]
    mpb.axis([-0.4, 0.4, -1, 1])
    mpb.xlabel("dy")
    mpb.ylabel("h/Hmax")
    ax = mpb.gca()  #  设置坐标轴

    U01, L01 = imagine(0.0001, L ,1)
    U02, L02 = imagine(0.0001, L, 2)
    U03, L03 = imagine(0.0001, L, 3) #  近轴成像
    for i in range(0, 1000, 1):      #  对U循环
        U=-Umax+du*i
        U1, L1 = imagine(U, L , 1)
        U2, L2 = imagine(U, L , 2)
        U3, L3 = imagine(U, L , 3)
        dy1.append( -(L01 - L1) * np.tan(U1))
        dy2.append( -(L02 - L2) * np.tan(U2))
        dy3.append( -(L03 - L3) * np.tan(U3))
        z.append( L*np.tan(U)/hmax)
    mpb.plot( dy1 ,z, color='red', linewidth=0.5,label='nc') #  修饰坐标并作图
    mpb.plot( dy2, z, color='blue', linewidth=0.5 ,label='nf')
    mpb.plot( dy3, z, color='green', linewidth=0.5 ,label='nd')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    mpb.grid(True)
    mpb.legend()
    mpb.show()   #  显示图像








