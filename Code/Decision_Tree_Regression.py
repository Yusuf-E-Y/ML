import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('bilet.csv')

x = data.sıra.values.reshape(-1,1)
y = data.fiyat.values.reshape(-1,1)

dt = DecisionTreeRegressor()

dt.fit(x,y)

yhead = dt.predict(x)

x1 = np.arange(min(x),max(x),0.1).reshape(-1,1)
yhead2 = dt.predict(x1)

plt.scatter(x,y)
plt.plot(x,yhead)
plt.plot(x1,yhead2)
plt.show()