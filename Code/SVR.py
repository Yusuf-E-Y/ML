import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = pd.read_csv("poly.csv")

sc = StandardScaler()

x = data.zaman.values.reshape(-1,1)
y = data.sicaklik.values

x1 = sc.fit_transform(x)
#y1 = sc.fit_transform(y)
y1 = sc.fit_transform(y.reshape(-1,1)).flatten()

sv = SVR(kernel="rbf")

sv.fit(x1,y1)

plt.scatter(x1,y1)
plt.plot(x1,sv.predict(x1),color="red") 
plt.show()