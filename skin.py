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
train_data=train.flow_from_directory("skin cancer\\train",
                                    batch_size=3,
                                    target_size=(250,250),
                                    class_mode='categorical')
validation_data=train.flow_from_directory("skin cancer\\test",
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
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy']
             )
q=model.fit(train_data,validation_data=validation_data,epochs=10)

print("Training Accuracy:", max(q.history["accuracy"]))
print("Validation Accuracy:", max(q.history["val_accuracy"]))

# Model Performance
predictions = model.predict(validation_data)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=validation_data.class_indices.keys()))

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(q.history['accuracy'], label='Training Accuracy')
plt.plot(q.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Model Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(q.history['loss'], label='Training Loss')
plt.plot(q.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Model Loss')
plt.legend()
plt.show()

model_json=model.to_json()

with open("model_architecture.json","w") as json_file:
    json_file.write(model_json)


model.save_weights("model_weights.weights.h5")
