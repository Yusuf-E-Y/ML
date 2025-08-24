import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

data = pd.read_csv("Datas/ürün.csv")

x = data.iloc[:,0:2].values
y = data.satinalma.values.reshape(-1,1)

S = data[data.satinalma==0]
B = data[data.satinalma==1]

xtrain,xtest,ytrain,ytest =  train_test_split(x,y,test_size=0.3,random_state=44)
Sc = StandardScaler()

xtrain1 = Sc.fit_transform(xtrain)
xtest1 = Sc.fit_transform(xtest) 

lr = LogisticRegression()
lr.fit(xtrain1,ytrain)

yhead = lr.predict(xtest1)
Cm = confusion_matrix(ytest,yhead)

print(Cm)
print(lr.score(xtest1,ytest))

plt.scatter(S.yaş,S.maaş,color="Blue")
plt.scatter(B.yaş,B.maaş,color="Red")
plt.show()