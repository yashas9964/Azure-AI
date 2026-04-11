import pandas as pd

data = pd.read_csv("WineQT.csv")
print(data.head())

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv("WineQT.csv")

# Create classification target
data['result'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

X = data.drop(['quality', 'Id', 'result'], axis=1)
y = data['result']

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X, y)
print("Decision Tree:", dt.score(X, y))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X, y)
print("Random Forest:", rf.score(X, y))

# SVM
svm = SVC()
svm.fit(X, y)
print("SVM:", svm.score(X, y))

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("PCA Output:\n", X_pca[:5])