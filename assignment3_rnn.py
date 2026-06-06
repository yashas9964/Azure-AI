# ==========================================================
# ASSIGNMENT 3
# RNN MODEL
# ==========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

print("="*70)
print("🔄 RECURRENT NEURAL NETWORK (RNN)")
print("="*70)

X = np.random.random((100,10,1))
y = np.random.randint(0,2,100)

rnn = Sequential([
    Input(shape=(10,1)),
    SimpleRNN(32),
    Dense(1,activation="sigmoid")
])

rnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

rnn.fit(
    X,
    y,
    epochs=3,
    verbose=0
)

print("\n✅ RNN Training Completed")
