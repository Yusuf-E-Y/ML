from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Datas/ürün.csv")

# Features (X): first 2 columns (age, salary)
x = data.iloc[:,0:2].values

# Target variable (y): purchase (0 or 1)
y = data.satinalma.values.reshape(-1,1)

# Class 0 (did not purchase)
S = data[data.satinalma == 0]

# Class 1 (purchased)
M = data[data.satinalma == 1]

# Visualize the two classes
plt.scatter(S.yaş, S.maaş, color="red")
plt.scatter(M.yaş, M.maaş, color="blue")

# Split the dataset: 67% train, 33% test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=23)

# Standardize features (mean=0, std=1)
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain)   # fit + transform on training data
xtest1 = sc.transform(xtest)         # only transform on test data

# Define SVM classifier (default kernel = RBF)
Sv = SVC(random_state=54)

# Train the model
Sv.fit(xtrain1, ytrain.ravel()) 

# Predict on test data
yhead = Sv.predict(xtest1)

# Evaluate: accuracy and confusion matrix
Total_score = Sv.score(xtest1,ytest), confusion_matrix(ytest,yhead)

# Print results
print(Total_score)

# Show plot
plt.show()
