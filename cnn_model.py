import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load Dataset
data_mild = os.listdir('DataSet_/MildDemented/')
data_moderate = os.listdir('DataSet_/ModerateDemented/')
data_non = os.listdir('DataSet_/NonDemented/')
data_verymild = os.listdir('DataSet_/VeryMildDemented/')

dot1 = []
labels1 = []

# Load Mild Demented Data
for img in data_mild:
    img_1 = cv2.imread('DataSet_/MildDemented/' + img)
    img_1 = cv2.resize(img_1, (50, 50))
    dot1.append(img_1)
    labels1.append(0)

# Load Moderate Demented Data
for img in data_moderate:
    img_2 = cv2.imread('DataSet_/ModerateDemented/' + img)
    img_2 = cv2.resize(img_2, (50, 50))
    dot1.append(img_2)
    labels1.append(1)

# Load Non-Demented Data
for img in data_non:
    img_3 = cv2.imread('DataSet_/NonDemented/' + img)
    img_3 = cv2.resize(img_3, (50, 50))
    dot1.append(img_3)
    labels1.append(2)

# Load Very Mild Demented Data
for img in data_verymild:
    img_4 = cv2.imread('DataSet_/VeryMildDemented/' + img)
    img_4 = cv2.resize(img_4, (50, 50))
    dot1.append(img_4)
    labels1.append(3)

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(dot1, labels1, test_size=0.2, random_state=101)
x_train = np.array(x_train)
x_test = np.array(x_test)

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)

# Define the Model
model = Sequential()

# Add Layers
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(50, 50, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(4, activation="softmax"))

# Compile the Model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the Model
history = model.fit(x_train, y_train_one_hot, batch_size=8, epochs=11 , verbose=1, validation_data=(x_test, y_test_one_hot))

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot, verbose=1)
print("-------------------------------------------------------------")
print("Performance Analysis ------> CNN")
print("-------------------------------------------------------------")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the Model
model.save('cnn_model_finetuned.h5')