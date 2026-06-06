# ==========================================================
# AZURE ASSIGNMENT - 2
# Decision Tree | Random Forest | SVM | PCA
# Reinforcement Learning | LSTM | Q-Network
# ==========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

print("=" * 70)
print("🚀 AZURE MACHINE LEARNING ASSIGNMENT")
print("=" * 70)

# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("WineQT.csv")

data["result"] = data["quality"].apply(
    lambda x: 1 if x >= 6 else 0
)

X = data.drop(["quality", "Id", "result"], axis=1)
y = data["result"]

print("\n✅ Dataset Loaded Successfully")
print("📊 Dataset Shape:", data.shape)

# ==========================================================
# DECISION TREE
# ==========================================================

from sklearn.tree import DecisionTreeClassifier

print("\n🌳 DECISION TREE")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

print("Accuracy:", round(dt.score(X, y) * 100, 2), "%")

# ==========================================================
# RANDOM FOREST
# ==========================================================

from sklearn.ensemble import RandomForestClassifier

print("\n🌲 RANDOM FOREST")

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

print("Accuracy:", round(rf.score(X, y) * 100, 2), "%")

# ==========================================================
# SVM
# ==========================================================

from sklearn.svm import SVC

print("\n📈 SUPPORT VECTOR MACHINE")

svm = SVC()
svm.fit(X, y)

print("Accuracy:", round(svm.score(X, y) * 100, 2), "%")

# ==========================================================
# PCA
# ==========================================================

from sklearn.decomposition import PCA

print("\n📊 PRINCIPAL COMPONENT ANALYSIS")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

print("First 5 PCA Rows:")
print(X_pca[:5])

# ==========================================================
# REINFORCEMENT LEARNING (Q-Learning)
# ==========================================================

print("\n🎮 REINFORCEMENT LEARNING")

states = 5
actions = 2

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9

for episode in range(100):

    state = np.random.randint(states)

    action = np.random.randint(actions)

    reward = np.random.randint(10)

    next_state = np.random.randint(states)

    Q[state, action] += alpha * (
        reward +
        gamma * np.max(Q[next_state]) -
        Q[state, action]
    )

print("Q Table:")
print(Q)

# ==========================================================
# LSTM MODEL
# ==========================================================

print("\n🧠 LONG SHORT-TERM MEMORY (LSTM)")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

X_lstm = np.random.random((100, 10, 1))
y_lstm = np.random.randint(0, 2, 100)

lstm_model = Sequential([
    Input(shape=(10, 1)),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

lstm_model.fit(
    X_lstm,
    y_lstm,
    epochs=3,
    verbose=0
)

print("✅ LSTM Training Completed")

# ==========================================================
# Q-NETWORK
# ==========================================================

print("\n🤖 Q-NETWORK")

q_network = Sequential([
    Input(shape=(4,)),
    Dense(24, activation="relu"),
    Dense(24, activation="relu"),
    Dense(2, activation="linear")
])

q_network.compile(
    optimizer="adam",
    loss="mse"
)

sample_state = np.random.random((1, 4))

prediction = q_network.predict(
    sample_state,
    verbose=0
)

print("Q-Network Prediction:")
print(prediction)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("🎉 ALL MODELS EXECUTED SUCCESSFULLY")
print("=" * 70)

print("""
✅ Decision Tree
✅ Random Forest
✅ SVM
✅ PCA
✅ Reinforcement Learning
✅ LSTM
✅ Q-Network
""")

print("=" * 70)