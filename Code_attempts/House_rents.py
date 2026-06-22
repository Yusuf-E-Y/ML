from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pandas as pd

dataFrame = pd.read_csv("kiraveri.csv")

X = dataFrame[['YIL','ODA SAYISI','M2']].values
y = dataFrame['ORT.FİYAT'].values.reshape(-1,1)

Rf = RandomForestRegressor(n_estimators=300,random_state=22) #Random Forest Algorithm

Rf.fit(X,y.ravel()) #Training
yhead = Rf.predict(X)

y, rc ,m2 = input().split()
new_data = [[int(y), int(rc), int(m2)]]
Pre = Rf.predict(new_data)
print(f"Prediction: {Pre[0]:.2f} TL")
