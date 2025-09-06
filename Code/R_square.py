import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Datas/maaş.csv")
lr = LinearRegression()

x = data.Tecrübe.values.reshape(-1,1)
y = data.Maaş.values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
lr.fit(x_train,y_train)
yhead = lr.predict(x_test)

plt.scatter(x,y)
plt.plot(x_test,yhead)

print(r2_score(y_test,yhead))