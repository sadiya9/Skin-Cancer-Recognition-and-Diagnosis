import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import os
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfile
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns



train=ImageDataGenerator(rescale=1/255)
train_data=train.flow_from_directory("skin cells\\Train",
                                    batch_size=3,
                                    target_size=(250,250),
                                    class_mode='categorical')
validation_data=train.flow_from_directory("skin cells\\Test",
                                    batch_size=3,
                           target_size=(250,250),
                           class_mode='categorical')

model = tf.keras.models.Sequential()

# Convolutional & Max Pooling layers
model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(250,250,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten & Dense layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))

# performing binary classification
model.add(tf.keras.layers.Dense(9, activation='softmax'))

model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy']
             )
res=model.fit(train_data,validation_data=validation_data,epochs=10)
print("Training Accuracy:", max(res.history["accuracy"]))
print("Validation Accuracy:", max(res.history["val_accuracy"]))

import matplotlib.pyplot as plt

# Predict on validation data
y_pred= np.argmax(model.predict(validation_data), axis=1)
y_true = validation_data.classes

# Confusion Matrix and Classification Report
print("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("Classification Report")
print(classification_report(y_true, y_pred))

# Plot Confusion Matrix as Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Oranges", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("CNN Confusion Matrix Heatmap")
plt.show()

# Overall Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Overall Accuracy:", accuracy)

# Plot Accuracy
plt.plot(res.history['accuracy'], label='Training Accuracy')
plt.plot(res.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(res.history['loss'], label='Training Loss')
plt.plot(res.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

model_json=model.to_json()

with open("model_architecture1.json","w") as json_file:
    json_file.write(model_json)


model.save_weights("model_weights.weights1.h5")
