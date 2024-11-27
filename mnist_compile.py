import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


# Define learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9  # Decay by 10% every epoch after 10 epochs


# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data and expand dimensions for CNN
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]  # Add channel dimension (28, 28, 1)
x_test = x_test[..., np.newaxis]

# Define CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ]
)
# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Callbacks
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
callbacks = [lr_reducer, early_stopping]

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2
)
datagen.fit(x_train)

# Train model with augmented data and learning rate scheduler
callbacks = [LearningRateScheduler(lr_schedule)]
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=15,
    callbacks=callbacks,
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model
model.save("mnist_cnn_model.keras")
