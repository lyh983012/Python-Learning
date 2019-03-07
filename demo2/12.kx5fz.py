from PIL import Image
import  random
import os

data = os.listdir('/Users/lyh/Desktop/data')
for i in data:
    ex=os.path.splitext(i)[1]
    if ex in ['.jgp', '.png', '.jpeg', '.bmp']:
        print(ex)
        frame=[]
        im = Image.open('/Users/lyh/Desktop/data/'+i)
        im = im.convert('RGB')
        im_red = im.copy()
        im_blue = im.copy()
        for k in range(30):

            move_x = int(random.uniform(0, k))
            move_y = int(random.uniform(0, k))

            imtemp = im.crop((move_x, move_y, im.width - move_x, im.height - move_y))
            im_redtemp = im_red.crop((2 * move_x, 2 * move_y, im.width, im.height))
            im_bluetemp = im_blue.crop((0, 0, im.width - 2 * move_x, im.height - 2 * move_y))

            r0, g0, b0 = imtemp.split()
            r1, g1, b1 = im_redtemp.split()
            r2, g2, b2 = im_bluetemp.split()

            new_im = [r1, g0, b2]
            print(new_im)
            imm = Image.merge('RGB', new_im)
            box1 = (5,5,380,380)  # 设置图像裁剪区域 (x左上，y左上，x右下,y右下)
            imm = imm.crop(box1)  # 图像裁剪
            frame.append(imm)

        print(frame)
        frame[0].save('/Users/lyh/Desktop/data/test.gif',format='GIF',append_images=frame[:],save_all=True,duration=1)
