import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import datetime
cap = cv2.VideoCapture("http://192.168.1.106:8081/?action=stream")
time.sleep(0.1)
#光流参数
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
color = np.random.randint(0,255,(100,3))

#初始化
ret=0
frame=[]
while not ret:
    print(1)
    ret,frame=cap.read()
ret,frame=cap.read()
#初始帧
term_crit = ( cv2.TERM_CRITERIA_EPS , 20, 1)
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask2 = np.zeros_like(frame)
#坐标
plt.axis([0, 10000, 0,10000])

x=5000
y=5000

fresh=0
while(1):
    fresh+=1
    begin = datetime.datetime.now()
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        if(p1 is None or p0 is None ):
            continue
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        list_of_vx=[]
        list_of_vy=[]
        #list_of_x=[]
        #list_of_y=[]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            #list_of_x.append(a)
            #list_of_y.append(b)
            list_of_vx.append(a-c)
            list_of_vy.append(b-d)
            mask2 = cv2.line(mask2, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        if len(list_of_vx)<5:

            #  # 丢失特征点，重绘初始帧 # #
            ret, frame = cap.read()
            term_crit = (cv2.TERM_CRITERIA_EPS, 20, 1)
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask2 = np.zeros_like(frame)
            continue
            # # # # # # # # # # # # # # # #

        #计算排除粗大误差
        #height=np.array(list_of_x).std()**2+np.array(list_of_y).std()**2
        #print('height=',height)
        array_of_vx=np.array(list_of_vx)
        array_of_vy=np.array(list_of_vy)
        mean_x=array_of_vx.mean()
        mean_y=array_of_vy.mean()
        std_x=array_of_vx.std()
        std_y=array_of_vy.std()
        count_point=0
        sum_x=0
        sum_y=0
        for i in range(len(array_of_vx)):
            vx=array_of_vx[i]
            vy=array_of_vy[i]
            if abs(vx-mean_x)<3*std_x and abs(vy-mean_y)<3*std_y:
                count_point+=1
                sum_x+=vx
                sum_y+=vy
        if count_point<5:

            # # 特征点混乱，重绘初始帧 # #
            ret, frame = cap.read()
            term_crit = (cv2.TERM_CRITERIA_EPS, 20, 1)
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask2 = np.zeros_like(frame)
            continue
            # # # # # # # # # # # # #

        vx=int(sum_x/count_point*10)
        vy=int(sum_y/count_point*10)
        end = datetime.datetime.now()
        k = (end - begin).microseconds/50000 #便于画图，实际可以计算准确像素速度
        x+=vx*k
        y+=vy*k
        plt.scatter(x,y,1)
        print('vx=',vx,' vy=',vy)
        im = cv2.add(frame, mask2)
        #图上作画
        cv2.imshow('im',im)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
        #更新角点
        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        print(x,y)
    else:
        break
plt.show()
