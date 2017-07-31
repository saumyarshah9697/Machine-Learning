import cv2
import numpy as np
import random
from tqdm import tqdm


def deshape(img):
    arr=[]
    for i in img:
        for x in np.array(i,dtype="uint8").tolist():
            print(x)
            arr.append(x)
    return arr

def reshape(arr,r,c):
    img1=[]
    for i in range(len(arr)):
        if i==0:
            temp=[]
        elif (i%c)==0:
            img1.append(temp)
            temp=[]
        temp.append(arr[i])        
    return img1

def norm(P1,P2):
    sum=0
    for (i,j) in zip(P1,P2):
        sum+=abs((i-j))
    return sum

def ClusterInit(arr,K):
    print("Generating ",K," Clusters from Arr")
    s="Generating "+str(K)+" Clusters from Arr\n"
   
    a=random.sample(arr,K)
   
    return a
    
    
def Compress(arr,Clusteroids,indexes):
   
    a=[]
    for i in indexes:
        a.append(Clusteroids[i])
       
    return a
    
def Closest(arr,Clusteroids):
    print("Computing Closest Clusteroids")
   
    indexes=[]
    count=1
    for i in tqdm(arr):
#         print("for ",count+1,"element out of 64")
        a="for "+str(count)+" element\n"
       
        temp =[]
        for j in Clusteroids:
            temp.append(norm(i,j))
        indexes.append(temp.index(min(temp)))
        
        count+=1
    print(indexes)
    return indexes

def ComputeMeans(arr,indexes,Clusteroids):
    newClus=[]
    print(len(arr))
    print(len(indexes))
    print(len(Clusteroids))
    for i in range(len(Clusteroids)):
        z=[]
        for j in indexes:
            if i == j:
                z.append(arr[indexes.index(j)])
        print(z)
        if len(z)==0:
            continue
        else:
            newClus.append(getmean(z))
    for a in newClus:
        
        if str(newClus)==str(Clusteroids):
            return ("end K Means",newClus)
    return (None,newClus)

def getmean(z):
    temp=[]
    for j in range(3):
        sum=0
        for i in range(len(z)):
            sum+=z[i][j];
        sum/=len(z)
        temp.append(int(sum))
    return temp
            

def Clusetering(arr,Clusteroids,iterations):
    for i in range(iterations):
        a=str(i)+"th Iteration\n"
        print(a)

        indexes=Closest(arr,Clusteroids)
      
        print("Computing means of clusteroids")
        a,Clusteroid=ComputeMeans(arr, indexes, Clusteroids)
        if(a=="end K means"):
            i=iterations
        Clusteroids=Clusteroid
    print("======================================================")    
    compressed_data=Compress(arr,Clusteroids,indexes)
    return compressed_data



# img=cv2.resize(cv2.imread("112.jpeg"),(100,100))
img=(cv2.imread("112.jpeg"))
arr=deshape(img)

K=100
iterations=5
Clusteroids=ClusterInit(arr, K)
print(arr[0])
print(Clusteroids)
data=Clusetering(arr, Clusteroids, iterations)
img2=reshape(data,img.shape[0], img.shape[1])
cv2.imshow("Original",cv2.resize(img,(500,500)))
cv2.imshow("Compressed",cv2.resize(np.array(img2,dtype="uint8"),(500,500)))

cv2.waitKey(0)