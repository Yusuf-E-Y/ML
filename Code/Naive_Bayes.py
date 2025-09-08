from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB #GaussianNB, BernoulliNB, MutlinomiaNB
"""
GaussianNB Tahmin edilen veri sürekli ise kullanılır
BernoulliNB Tahmin edilen veri ikili ise kullanılır
MutlinomiaNB integer sayılardan oluşuyorsa kullanılır
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Datas/ürün.csv")

x = data.iloc[:,0:2].values
y = data.satinalma.values.reshape(-1,1)

S = data[data.satinalma == 0]
M = data[data.satinalma == 1]

plt.scatter(S.yaş,S.maaş,color="red")
plt.scatter(M.yaş,M.maaş,color="blue")

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=23)
sc = StandardScaler()

xtrain1 = sc.fit_transform(xtrain)
xtest1 = sc.transform(xtest)

nb = GaussianNB()
nb.fit(xtrain1,ytrain.ravel())

yhead = nb.predict(xtrain1)

nb.score(xtest1,ytest)