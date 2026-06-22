import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("Datas/bilet.csv")

x = data.sıra.values.reshape(-1,1)
y = data.fiyat.values.reshape(-1,1)

rf = RandomForestRegressor(n_estimators=100,random_state=22)

rf.fit(x,y)
yhead = rf.predict(x)

plt.scatter(x,y)
plt.plot(x,rf.predict(x))

x1 = np.arange(min(x),max(x),0.01).reshape(-1,1)
yhead1 = rf.predict(x1)

plt.scatter(x,y)
plt.plot(x1,rf.predict(x1))
plt.show()