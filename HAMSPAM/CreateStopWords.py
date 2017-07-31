# coding= utf-8

from nltk.tokenize import sent_tokenize, word_tokenize
import re
from tqdm import tqdm
from nltk.stem import PorterStemmer
ps=PorterStemmer()

data="HamSpamData.txt"


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
    m=re.sub("[\w+]\.*.+\.[\w]{2,3}[[?/=]*[\w*]]*"," website",m)
    m=re.sub("[\w\.-]+@[\w\.-]+"," email",m)
    m=re.sub(u'[$��]+'," money",m)
    m=re.sub("'<[^<>]+>'","",m)
    m=re.sub("[0-9]+"," number",m)
    m=ps.stem(m)
    return m



def Create_word_list(data):
    #first of all, clean the data
    dict={}
    f1=open(data)
    f2=open("TempDump.txt","w")
    unc=f1.readlines()
    unc=Preprocessing(unc)
    print(len(unc))
    for x in unc:
        for j in x[1]:
            f2.write(j+"\n")
    f2.close()        
    f2=open("TempDump.txt","r")
    X=word_tokenize(f2.read())
    for word in X:
        if re.match("[\w|\d]{3,2000}", word):
            count=dict.get(word,0)
            dict[word]=count+1    
    f2.close()
    wrd=sorted(dict,key=dict.get,reverse=True)
    f2=open("StopWords.txt","w")
    for i in range(200):
        print(wrd[i],dict[wrd[i]])
        f2.write(str(i)+","+wrd[i]+"\n")
    
Create_word_list(data)
