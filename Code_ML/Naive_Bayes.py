from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # GaussianNB, BernoulliNB, MultinomialNB
"""
GaussianNB: Use when the predicted data is continuous
BernoulliNB: Use when the predicted data is binary
MultinomialNB: Use when the data consists of integer counts
"""
import pandas as pd
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv("Datas/ürün.csv")

# Features (X): first 2 columns (age, salary)
x = data.iloc[:,0:2].values

# Target variable (y): purchase (0 or 1)
y = data.satinalma.values.reshape(-1,1)

# Separate class 0 (did not purchase) and class 1 (purchased) for plotting
S = data[data.satinalma == 0]
M = data[data.satinalma == 1]

# Scatter plot to visualize the two classes
plt.scatter(S.yaş, S.maaş, color="red")
plt.scatter(M.yaş, M.maaş, color="blue")

# Split dataset: 67% training, 33% testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=23)

# Standardize features (mean=0, std=1)
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain)   # fit + transform on training data
xtest1 = sc.transform(xtest)         # transform test data using the same scaler

# Define Gaussian Naive Bayes classifier
nb = GaussianNB()

# Train the model on training data
nb.fit(xtrain1, ytrain.ravel())

# Predict labels for training data
yhead = nb.predict(xtrain1)

# Evaluate model on test data (accuracy)
nb.score(xtest1, ytest)
