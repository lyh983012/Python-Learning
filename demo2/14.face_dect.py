import numpy as np
import cv2
import cvlib as cv


def tec_face():
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        faces, confidences = cv.detect_face(frame)
        if(faces!= [] ):
            c, r, w, h = faces[0]
            if not(c > 0 and r > 0 and w > 0 and h > 0):
                continue
            break
    c, r, w, h = faces[0][0], faces[0][1], faces[0][2] - faces[0][0], faces[0][3] - faces[0][1]
    print(r,r+h,c,c+w)
    return (c, r, w, h , frame)


#可以改成树莓派的
cap = cv2.VideoCapture(0)
# 角点检测的参数，第一个是角点个数限制，第二个是角点的显著程度（0.1～1)，第三个是角点间距限制，第四个是使用的周围的点数（影响识别速度和准确度）
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )
# 光流的参数，不知道什么意思
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

# 几个用来画图的随机颜色
color = np.random.randint(0,255,(100,3))

#更改成为角点检测？ 使用光信号是否可以？（XXXXX）

c,r,w,h,frame=tec_face() #自己def的函数，调用cvlib识别人脸，这里是检测人脸的四角，并且输出这一帧
                        #后面的函数是检测这个矩形内的物体的移动

color = np.random.randint(0,255,(100,3))
track_window = (c,r,w,h) #创建初试窗口

#创建矩形

roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#生成蒙版
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# 设置尺度
term_crit = ( cv2.TERM_CRITERIA_EPS , 20, 1)

#旧的，p0是角点
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask2 = np.zeros_like(frame)

fresh=0
while(1):
    fresh+=1
    if(fresh==30):
        c, r, w, h, frame = tec_face()
        track_window = (c, r, w, h)  # 创建初试窗口
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        fresh=0

    ret ,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 使用meanshift算法调整矩形大小（核心，追踪人脸）
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # 寻找焦点的光流变化（次要，光流计算）
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
        if(p1 is None or p0 is None):
            continue
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        #画出轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask2 = cv2.line(mask2, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        im = cv2.add(frame, mask2)

        #图上作画
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        im = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('im',im)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        #更新角点
        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    else:
        break

cv2.destroyAllWindows()
cap.release()