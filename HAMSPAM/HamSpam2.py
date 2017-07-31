from nltk.tokenize import sent_tokenize, word_tokenize
import re
from tqdm import tqdm
from nltk.stem import PorterStemmer
from sklearn.neural_network import MLPClassifier
ps=PorterStemmer()

f2=open("Model3.txt","w")

''' Input:[...........,sents1,sents2,sents3,.........]
    Output:[.......[0,[sentdata]],.........[1,[sentdata]].......]
'''
def Preprocessing(db):   
    X=[]
    for i in tqdm(db):
        a=re.findall("(spam|ham)\s+(.+)",i)
        X.append(Spam_Ham(a))
    return X


''' Input:[(spam,sents)]
    Output:[0,[sentdata]]
'''
def Spam_Ham(i):
    j=[]
    if i[0][0]=="spam":
        j.append(1)
    if i[0][0]=="ham":
        j.append(0)
    j.append(Sent_Data(i[0][1]))
    return j

''' Input:sents
    Output:[.....cleansent1,cleansent2,.......]
'''
def Sent_Data(x):
    Xi=sent_tokenize(x, "english")
    J=[]
    for m in Xi:
        J.append(Clean_Process(m))
    return J


''' Input:sent
    Output:cleansent
'''    
def Clean_Process(m):
    m=m.lower()
    m=re.sub("[w]{3}\..+\.[\w]{2,3}[/*\w+]*"," website",m)
    m=re.sub("[\w\.-]+@[\w\.-]+"," email",m)
    m=re.sub(u'[$€£]+'," money",m)
    m=re.sub("'<[^<>]+>'","",m)
    m=re.sub("[0-9]+"," number",m)
    m=ps.stem(m)
    return m



''' Input:dataVec,[0|1,[....,....,.....]]
    Output:0|1,[Vec of size data Vec filled with 1|0]
'''    
def Vectorize(vec,a):
    X=[0]*len(vec)
    temp=[]
    for m in a: 
#         print(m)
        m=sent_tokenize(m,"english")
        for n in m:
            for j in word_tokenize(n):
                j=ps.stem(j)
                if j in vec:
                    temp.append(vec.index(j))
    for j in temp:
        X[j]=1
    return X  


def CreateDataSet():
    f1=open("HamSpamData.txt")
    db=f1.readlines()
    a=Preprocessing(db)
    DataVector=[]
    for i in open("StopWords.txt","r").readlines():
        DataVector.append(i.split(",")[1].rstrip())
    Xtrain=[]
    ytrain=[]
    for b in tqdm(a[:5000]):
        f2.write(str(b)+"\n")
        xyz=Vectorize(DataVector,b[1])
        Xtrain.append(xyz)
        ytrain.append([b[0]])
        print(b[0])
    xtest=[]
    for b in tqdm(a[4001:]):
        xyz=Vectorize(DataVector,b[1])
        xtest.append(xyz)
    return(Xtrain,ytrain,xtest)

def Mtrain(X,y):
    model = MLPClassifier(alpha=1e-5,activation="logistic",hidden_layer_sizes=(1000))
    model.fit(X,y)
    return model


def Predictor(X,Model):
    Y=[]
    for a in tqdm(X):
        Y.append(Model.predict(a))
    return Y
#     return Model.predict(X)
# for b in tqdm(a[5000:]):

f3=open("TempDump.txt","w")
X=CreateDataSet()
model=Mtrain(X[0],X[1])
Y=Predictor(X[2],model)
for i in range(len(Y)):
    f3.write(str(i+4001)+" "+str(Y[i])+"\n")