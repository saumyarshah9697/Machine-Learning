import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import timeit
import pickle
#import msvcrt as m
Font=cv2.FONT_HERSHEY_PLAIN


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
    img=cv2.resize(img,(50,50))
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
        img=cv2.imread(x2)
        cv2.putText(img,predn,(50,50), Font,1, (255,255,255),2, cv2.LINE_AA)
        cv2.imshow("Frame",img)
        cv2.waitKey(0)
#        wait()
# 
# model=svm.SVC()
start = timeit.default_timer()
X,y=createTrainSet()
stop = timeit.default_timer()
f1.write(str(stop-start))
print(str(stop-start))
# model.fit(X,y)
# GenerateTestResults(model)

clf = MLPClassifier(alpha=1e-5,activation="logistic",hidden_layer_sizes=(22500))
start = timeit.default_timer()
clf.fit(X,y)
stop = timeit.default_timer()
f1.write(str(stop-start))
print(stop-start)
start = timeit.default_timer()
pickle.dumps(clf)
GenerateTestResults(clf)
stop = timeit.default_timer()
f1.write(str(stop-start))
print(stop-start)