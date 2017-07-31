import cv2
import os
from sklearn import svm
import math
import numpy as np
from tqdm import tqdm
#import msvcrt as m


#def wait():
#    m.getch()

f1=open("Models.txt","w")
f2=open("Models2.txt","w")

train="images/Train"
test="images/Test"

def label_img(img):
    a=int(img.split(".")[0])
    if a<184:
        a=str(img+" No Goal \n")
        f1.write(a)
        return 0
    else:
        a=str(img+" Goal \n")
        f1.write(a)
        return 1
    
def getNPData(img):
    img=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(500,500))
    a=np.array(img);
    temp=[]
    for x in a:
        for y in x:
            temp.append(y)
    return temp

def createTrainSet():
    training_setX=[]
    training_setY=[]
    for x in tqdm(os.listdir(train)):
        x2=os.path.join(train,x)
        y=label_img(x)
        X=getNPData(x2)
        training_setX.append(X)
        training_setY.append(y)
#    np.save('train_data.npy', [training_setX,training_setY])
    return (training_setX,training_setY)

def GenerateTestResults(model):
    a=os.listdir(test)
    print(a)
    for x in tqdm(a):
        x2=os.path.join(test,x)
        pred=model.predict(getNPData(x2))
        if pred==0:
            a=str(x+" No Goal \n")
            f2.write(a)
            predn="NO GOAL"
        else:
            a=str(x+"Goal \n")
            f2.write(a)
            predn="GOAL"
        print(x2+" "+predn)
#        cv2.imshow(predn,x)
#        wait()


model=svm.SVC()
X,y=createTrainSet()
model.fit(X,y)
GenerateTestResults(model)