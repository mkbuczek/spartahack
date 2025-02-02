import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.datasets import mnist

#load EMNIST letters dataset
train_df = pd.read_csv('emnist-letters-train.csv', header=None)
test_df = pd.read_csv('emnist-letters-test.csv', header=None)

#load the mapping file to convert labels to letters
with open('emnist-letters-mapping.txt', 'r') as f:
    mapping = {int(line.split()[0]): line.split()[1] for line in f.readlines()}

#split the data into features and labels
X_train = train_df.iloc[:, 1:].values  
y_train = train_df.iloc[:, 0].values  

X_test = test_df.iloc[:, 1:].values  
y_test = test_df.iloc[:, 0].values

#normalize the image data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#reshape the data to match the input shape for the model (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

#ensure the labels range from 0 to 25 (A-Z)
y_train = y_train - 1
y_test = y_test - 1

y_train = to_categorical(y_train, 26) 
y_test = to_categorical(y_test, 26)    

#create the model
def create_model():
    model = Sequential()

    #first convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    
    #second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    #flatten the data for fully connected layers
    model.add(Flatten())
    
    #fully connected layer
    model.add(Dense(128, activation='relu'))
    
    # Dropout for regularization
    model.add(Dropout(0.5))
    
    # Output layer with 26 units (for 26 letters) and softmax activation
    model.add(Dense(26, activation='softmax'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(

    rotation_range=5,      # Rotate images up to 10 degrees
    width_shift_range=0.05,  # Shift images horizontally by 10% of width
    height_shift_range=0.05, # Shift images vertically by 10% of height
    shear_range=0.05,        # Shear transformation
    zoom_range=0.05,         # Random zoom
    horizontal_flip=False,  # Flip images (set to False for EMNIST letters)
    fill_mode='nearest'     # Fill pixels with nearest value
)

#create and train the model
model = create_model()
# model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
model.fit(datagen.flow(X_train, y_train, batch_size=64), 
          epochs=30, 
          validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

#predict function to map model output to letter
def predict_image(model, image_data):
    #preprocess the image (assuming it's a 28x28 image)
    image_data = image_data.astype('float32') / 255.0
    image_data = image_data.reshape(1, 28, 28, 1)  # Add batch dimension
    
    #predict the class (0-25)
    prediction = model.predict(image_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    predicted_letter = mapping[predicted_class + 1]
    return predicted_letter

#save the trained model
model.save('emnist_letter_model.keras')