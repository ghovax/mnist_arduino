import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(11, activation="softmax"))
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
)
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)

model.save("mnist_model.keras")