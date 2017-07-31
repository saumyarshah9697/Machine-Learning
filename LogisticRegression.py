import math
import sys

def HypoThe(Theta,xi):
    if(len(Theta)==len(xi)):
        sum=0
        for i in range(len(xi)):
            sum+=(Theta[i]*xi[i])
        return Sigmoid(sum)
    else:
        return False

def Sigmoid(A):
    return 1/(1+math.exp(A))



# def RegCostFunc(Theta,X,Y,lamb):
#     sum1=0
#     for i in range(len(X)):
#         sum1+=((HypoThe(Theta, X[i])-Y[i])**2)
#     J=sum1
#     sum1=0
#     for i in range(1,len(Theta)):
#         sum1+=(Theta[i]**2)
#     J+=lamb*sum1
#     return J/(2*len(X))
#     


# def CheckAccuracy(Theta,X,Y):
#     err=0.00000
#     for i in range(len(X)):
#         err=err+math.fabs((HypoThe(Theta, X[i])-Y[i])/Y[i])
#     err=err/len(X)
#     print("M.S.E is ",1-err)
#     

def Predict(Theta,X):
    a=HypoThe(Theta, X)
    if(a<0.5):
        return 0
    else:
        return 1

def GradTerm(X,Y,Theta,i):
    sum1=0
    for j in range(len(X)):
        sum1+=(((-Y[i]*math.log(HypoThe(Theta,X[i])))-((1-Y[i])*math.log(1-HypoThe(Theta,X[i]))))*(X[j][i]))
    return sum1
    
        
def GradDesc(Theta,alpha,Xfeature,Ylabels,lamb):
    Theta_=[]
    Theta_.append(Theta[0]-(GradTerm(Xfeature,Ylabels,Theta,0)*alpha/len(Xfeature)))
    for i in range(1,len(Theta)):
        Theta_.append((Theta[i]*(1-((alpha*lamb)/len(Xfeature))))-(alpha*(GradTerm(Xfeature,Ylabels,Theta,i))/len(Xfeature)))
    return Theta_
    
    

def LinearRegression(Xfeature,Ylabels,alpha,lamb,iterations):
    if len(Xfeature)!=len(Ylabels):
        print("Missing Data");
        return False
    else:   
        Theta=[]
        for i in range(len(Xfeature)):
            Xfeature[i].insert(0,1)
        for i in range(len(Xfeature[i])):
            Theta.append(0)
        print("========================================================================================")
        
        for i in range(iterations):
            print("\nIteration Number ",i)
            print(Theta)
            
            Theta=GradDesc(Theta, alpha, Xfeature, Ylabels, lamb)
#             Cost=RegCostFunc(Theta, Xfeature, Ylabels, lamb)
            print(Theta)

            print("========================================================================================")
        
        print(Theta)
        return Theta
            
Xfeature=[] 
Ylabels=[]   
f1=open("LogisticRegression.txt")
z=f1.readline()
print("Fetching Data")
while z:
    print(".",end=".")
    temp=z.split(",")
    temp1=[]
    for i in range(len(temp)-1):
        temp1.append(float((temp[i])))
    Xfeature.append(temp1)
    Ylabels.append(float(temp[-1]))
    z=f1.readline()
f1.flush()
print("")
iterations=int(input("Enter the Number of Iterations"))
alpha=float(input("Enter the learning rate"))
lamb=float(input("Enter the regularization term"))
m=round(len(Xfeature)/5)
print(Xfeature[0]," ", Ylabels[0]," ", alpha," ", lamb," ", iterations )


Theta=LinearRegression(Xfeature[:(4*m)], Ylabels[:(4*m)], alpha, lamb, iterations)

# CheckAccuracy(Theta, Xfeature[(4*m):], Ylabels[(4*m):])

print("\n========================================================================================")

print("Saving Model......")   
f2=open("Models.txt","a")
f2.write(str(Theta)+"\n")
f2.close
print("========================================================================================\n")
while True:
    print("Enter ",len(Xfeature[0])," features to predict value for")
    Pred=[1];
    sys.stdout.flush()
    for i in range(len(Xfeature[0])-1):
        temp=float(input("Enter feature number "))
        Pred.append(temp)
    print(Predict(Theta,Pred))