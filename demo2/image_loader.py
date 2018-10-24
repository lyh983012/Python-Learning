from PIL import Image
import struct


def read_image(filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    dataset = []
    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    for i in range(images):
        temp=[]
        for x in range(rows):
            for y in range(columns):
                temp.append(int(struct.unpack_from('>B', buf, index)[0])/256.0)
                index += struct.calcsize('>B')
        dataset.append(temp)
    return dataset

def read_label(filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labelArr=[]
    n=0
    for x in buf:
        n+=1
        if (n>8 and n<9694):
            temp=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            temp[x]=0.8
            labelArr.append(temp)
    return labelArr,n


if __name__ == '__main__':
    read_image('t10k-images-idx3-ubyte')
    read_label('t10k-labels-idx1-ubyte')