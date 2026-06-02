# ==========================================================
# ASSIGNMENT 3
# CNN | COMPUTER VISION | IMAGE PROCESSING
# IMAGE CLASSIFICATION
# ==========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

print("="*70)
print("🖼️ CNN | COMPUTER VISION | IMAGE CLASSIFICATION")
print("="*70)

# ==========================================================
# COMPUTER VISION
# ==========================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("\n✅ Images Loaded Successfully")
print("Training Images:", x_train.shape)

# ==========================================================
# IMAGE PROCESSING
# ==========================================================

x_train = x_train.reshape(-1,28,28,1)/255.0
x_test = x_test.reshape(-1,28,28,1)/255.0

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

print("✅ Image Processing Completed")

# ==========================================================
# CNN MODEL
# ==========================================================

cnn = Sequential()

cnn.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)

cnn.add(MaxPooling2D((2,2)))

cnn.add(Flatten())

cnn.add(Dense(128,activation="relu"))

cnn.add(Dense(10,activation="softmax"))

cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

cnn.fit(
    x_train,
    y_train,
    epochs=2,
    batch_size=64,
    verbose=0
)

loss, accuracy = cnn.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n🎯 Image Classification Accuracy:")
print(round(accuracy*100,2), "%")

print("\n✅ CNN Completed")
print("✅ Computer Vision Completed")
print("✅ Image Processing Completed")
print("✅ Image Classification Completed")