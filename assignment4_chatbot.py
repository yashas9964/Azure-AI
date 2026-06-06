# ==========================================================
# ASSIGNMENT 4
# CHATBOT USING RNN AND LSTM
# ==========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("="*70)
print("🤖 CHATBOT USING LSTM")
print("="*70)

# ==========================================================
# TRAINING DATA
# ==========================================================

questions = [
    "hello",
    "hi",
    "how are you",
    "what is your name",
    "bye"
]

answers = [
    "Hello User",
    "Hi There",
    "I am Fine",
    "I am an AI Chatbot",
    "Goodbye"
]

# ==========================================================
# TEXT PROCESSING
# ==========================================================

tokenizer = Tokenizer()

tokenizer.fit_on_texts(questions)

X = tokenizer.texts_to_sequences(questions)

X = pad_sequences(X, maxlen=4)

y = np.arange(len(answers))

# ==========================================================
# LSTM MODEL
# ==========================================================

model = Sequential()

model.add(
    Embedding(
        input_dim=50,
        output_dim=8,
        input_length=4
    )
)

model.add(
    LSTM(32)
)

model.add(
    Dense(
        len(answers),
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

print("✅ LSTM Chatbot Trained Successfully")

# ==========================================================
# CHATBOT
# ==========================================================

print("\nType 'exit' to stop")

while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "exit":
        print("Bot : Goodbye")
        break

    seq = tokenizer.texts_to_sequences(
        [user_input]
    )

    seq = pad_sequences(
        seq,
        maxlen=4
    )

    prediction = np.argmax(
        model.predict(seq, verbose=0)
    )

    print(
        "Bot :",
        answers[prediction]
    )