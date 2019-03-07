
import csv

filename = 'text.csv'
maxpeople=20
traindata = []
trainlable = []
name=["Me","DPI","CANMOVE","DAE","FLY"]
myread=[]

with open(filename) as f:
    reader = csv.reader(f)

    for row in reader:
        myread.append(row)
    for sj in range(4):
        mylist=[]
        for row1 in range(5):
            mylist.append([])
        for j in range(5):
            jl=0
            for row in myread:
                if int(row[12+sj])==1:
                    for i in range(5):
                        if int(row[16+i])==j+1 and len(mylist[i])<maxpeople:
                            mylist[i].append(row)
                            myread[jl][12+sj]='0'
                            myread[jl][16+i]='-2'
        print('',sj+1,'\n')
        for i in range(5):
            print(name[i])
            print(test)
            print('\n\n')


