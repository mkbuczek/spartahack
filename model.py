import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

# Load the EMNIST dataset (train and test)
train_data = pd.read_csv("3/emnist-byclass-train.csv", header=None)
test_data = pd.read_csv("3/emnist-byclass-test.csv", header=None)

# Separate the features (images) and labels
x_train = train_data.iloc[:, 1:].values  # all rows, columns from 1 to end (pixel values)
y_train = train_data.iloc[:, 0].values   # first column (labels)
x_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Reshape the images to 28x28
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# The number of classes in EMNIST
num_classes = 8144

# Create the model
def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = create_model()

# Train the model with sparse labels (no need to one-hot encode labels)
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save("emnist_model.keras")