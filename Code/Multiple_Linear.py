import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('reklam.csv')

x = data.iloc[:,1:4].values
y = data.satış.values.reshape(-1,1)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=22)
lr = LinearRegression()

lr.fit(x,y)

yhead = lr.predict(xtest)

print(lr.predict([[230,38,70]])[0][0])
