import cv2
import os

w=20
h=20

drawing = False #鼠标按下为真
mode = True #如果为真，画矩形，按m切换为曲线



def draw_rac(event,x,y,flags,param):
    global ix,iy,ix2,iy2,next,img

    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy=x,y

    elif event == cv2.EVENT_LBUTTONUP:
        ix2, iy2 = x, y
        img1 = img.copy()
        cv2.rectangle(img1, (ix, iy), (ix2, iy2), (0, 255, 0), 1)
        cv2.imshow("camera", img1)
        next = 1


if __name__ == '__main__':
    global img,ix,iy,ix2,iy2
    ix, iy, ix2, iy2=(0,0,0,0)
    n=0
    f = open('/Users/lyh/Desktop/train/bg/bg.txt' , "w+")
    
    for filename in os.listdir('/Users/lyh/Desktop/train/NEG/'):  # bg image 批处理
        #print(filename)
        if filename.find('jpg')==-1:
            if filename.find('png') == -1:
                if filename.find('jpeg') == -1:
                    continue
        try:
            img = cv2.imread('/Users/lyh/Desktop/train/NEG/'+filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (4*w, 4*h))
            cv2.imwrite('/Users/lyh/Desktop/train/bg/' + str(n + 1).rjust(3, '0') + '.jpg', img)

            print(n)
            f.write('/Users/lyh/Desktop/train/bg/' + str(n + 1).rjust(3, '0') + '.jpg'+ "\n")
            n += 1
        except KeyboardInterrupt:
            print('暂停一下')
    f.close()
    n=0
    '''
    n = 0
    for filename in os.listdir('/Users/lyh/Desktop/train/POS/'):  # bg image 批处理
        print(filename)
        if filename.find('jpg') == -1:
            if filename.find('png') == -1:
                if filename.find('jpeg') == -1:
                    continue
        try:
            img = cv2.imread('/Users/lyh/Desktop/train/POS/' + filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (w, h))
            cv2.namedWindow("camera", 0)
            cv2.setMouseCallback("camera", draw_rac)
            cv2.imshow("camera", img)
            k = cv2.waitKey(-1) & 0xFF
            if (k == ord('n')) :
                cv2.imwrite('/Users/lyh/Desktop/train/hd/' + str(n + 1).rjust(3, '0') + '.jpg', img)
                print(ix,iy,ix2,iy2)
                f.write('bg/' + str(n + 1).rjust(3, '0') + '.jpg'+' '+str(1)+' '+str(ix)+' '+str(iy)+' '+str(ix2-ix)+' '+str(iy2-iy)+"\n")
                n += 1
                continue

        except KeyboardInterrupt:
            print('暂停一下')
    f.close()
    '''

