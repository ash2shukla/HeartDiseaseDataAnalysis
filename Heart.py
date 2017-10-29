import numpy as np
from sklearn.model_selection import train_test_split
from pyswarm import pso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class psoKNN:
    def __init__(self,path):
        data = np.genfromtxt(path,delimiter=',')
        data = data[~np.any(np.isnan(data),axis=1)]
        X,Y=data[:,:-1],data[:,-1:]
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
        self.Y_train = self.Y_train.reshape(1,self.Y_train.shape[0])[0]
        self.Y_test = self.Y_test.reshape(1,self.Y_test.shape[0])[0]
        # Scale X to improve accuracy
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train =scaler.transform(self.X_train)
        scaler.fit(self.X_test)
        self.X_test =scaler.transform(self.X_test)      
    def minKNN(self,n):
        n = int(n+0.5)
        model = KNeighborsClassifier(n)
        model.fit(self.X_train,self.Y_train)
        return 1-model.score(self.X_test,self.Y_test)
        

if __name__ == "__main__":
    inst = psoKNN('processed.txt')
    resp = pso(inst.minKNN,[1],[198],maxiter=1,debug=True)
    res = []
    for i in range(1,198):
        res.append(inst.minKNN(i))
    print min(res)
    plt.plot(res)
    plt.plot(res.index(min(res)),min(res),marker='o',label=str(min(res)))
    print "neighbors can't be fractional"
    print "Checking accuracy is neighborhood i.e. %d %d %d"%(int(resp[0])-1,int(resp[0]),int(resp[0])+1)
    neigh= []
    neigh.append(inst.minKNN(int(resp[0])-1))
    neigh.append(inst.minKNN(int(resp[0])))
    neigh.append(inst.minKNN(int(resp[0])+1))
    print "Least Error found at %d"%(int(resp[0])-1+neigh.index(min(neigh)))
    plt.show()
