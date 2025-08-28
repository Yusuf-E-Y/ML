from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Datas/ürün.csv")

x = data.iloc[:,0:2].values
y = data.satinalma.values.reshape(-1,1)

S = data[data.satinalma==0]
B = data[data.satinalma==1]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=33)
sc = StandardScaler() 

xtrain1 = sc.fit_transform(xtrain)
xtest1 = sc.transform(xtest)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(xtrain1,ytrain)
yhead = knn.predict(xtest1)
#print(knn.score(xtest1,ytest))

scoreList = list()

COUNT = 21
for i in range(1,COUNT):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain1,ytrain)
    yhead = knn.predict(xtest1)

    generations = knn.score(xtest1,ytest)
    scoreList.append(generations)

scoreList.sort()

for index,j in enumerate(scoreList, start=1):
    print(index, j)

cn = confusion_matrix(ytest,xtest1)
print(cn)

plt.plot(range(1,COUNT), scoreList)
plt.show()
    