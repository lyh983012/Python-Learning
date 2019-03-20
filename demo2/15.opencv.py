import numpy as np
import cv2
import cvlib as cv
import datetime
import random
import time


n1=0
framecount=30 #调整间隔多少帧进行一次识别
fresh=0
term_crit = ( cv2.TERM_CRITERIA_EPS , 20, 1) #人脸识别迭代上线以及精度

def find_hand(frame):

    global n1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hand_cascade = cv2.CascadeClassifier('/Users/lyh/Desktop/挑战杯/Opencv-master/haarcascade/cascade.xml')
    hands=hand_cascade.detectMultiScale(gray,1.1,10,cv2.CASCADE_SCALE_IMAGE)
    print(hands)
    (c, r, w, h)=(0,0,0,0)
    for hand in hands:
        (c, r, w, h)=hand
        x=c
        y=r
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if hands is not []:
        return (c, r, w, h , frame)
    else:
        return []

def find_face(frame):

    faces, confidences = cv.detect_face(frame)
    max=0
    count=0
    for i in range(len(confidences)):
        if confidences[i]>max:
            count=i
            max=confidences[i]
    if faces:
        c, r, w, h = faces[count]
        if (c > 0 and r > 0 and w > 0 and h > 0):
            c, r, w, h = faces[count][0], faces[count][1], faces[count][2] - faces[count][0], faces[count][3] - faces[count][1]
            print(r,r+h,c,c+w)
            return (c, r, w, h , frame)
    else:
        return []

cap = cv2.VideoCapture(0)


while(1):
    i = random.randint(1, 20)
    print(i)
    frame=cv2.imread('/Users/lyh/Desktop/train/POS/n'+str(i)+'.jpg')
    ret ,frame2222 = cap.read()
    begin = datetime.datetime.now()
    if( fresh == 0 ):
        temp = find_hand(frame)
        if not temp :
            continue
        c, r, w, h, frame = temp
        track_window = (c, r, w, h)  # 创建初试窗口
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        fresh=1
    #fresh += 1
    fresh=0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 使用meanshift算法调整矩形大小（核心，追踪人脸）
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        c, r, w, h=track_window
        #计算坐标
        s = w * h
        x,y=(int(c + w/ 2), int(r + h/ 2))
        print("S=",s,"X=",x,"Y=",y)
        cv2.circle(frame, (int(c + w/ 2), int(r + h/ 2)), 10, (0, 255, 0), -1)
        #图上作画，运行版本可以去掉
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.namedWindow("camera", 0)
        im = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow("camera",im)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
    if(fresh == framecount):
        fresh+=0
    time.sleep(1)
    end = datetime.datetime.now()
    k=end - begin
    print('time_cost:=',k.microseconds/1000,'ms')



cv2.destroyAllWindows()
cap.release()