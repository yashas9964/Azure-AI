import pandas as pd

data = pd.read_csv("WineQT.csv")
print(data.head())

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("WineQT.csv")

# Features & Target
X = data.drop(['quality', 'Id'], axis=1)
y = data['quality']

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
print("Prediction:", model.predict(X.iloc[[0]]))

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("WineQT.csv")

# Convert quality into good(1)/bad(0)
data['result'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

X = data.drop(['quality', 'Id', 'result'], axis=1)
y = data['result']

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Accuracy:", model.score(X, y))