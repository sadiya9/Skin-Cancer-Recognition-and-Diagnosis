import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns




train=ImageDataGenerator(rescale=1/255)
train_data=train.flow_from_directory("skin cells\\Train",
                                    batch_size=3,
                                    target_size=(250,250),
                                    class_mode='categorical')
validation_data=train.flow_from_directory("skin cells\\Test",
                                    batch_size=3,
                                    target_size=(250,250),class_mode='categorical')





def residual_block(x, filters, kernel_size=3, stride=1):
    # Shortcut connection
    shortcut = x

    # First convolutional layer
    x = Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    # Add shortcut connection
    if stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x






def build_mlp(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    x = residual_block(x, filters=64, stride=1)
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=512, stride=2)

    # Classification head
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Create model
    model = Model(inputs, outputs)
    return model





model = build_mlp((250,250,3),9)


model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
l = model.fit(train_data,validation_data=validation_data,epochs=10)
print(max(l.history["accuracy"]))

print("Training Accuracy:", max(l.history["accuracy"]))
print("Validation Accuracy:", max(l.history["val_accuracy"]))

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
sns.heatmap(cm, annot=True, cmap="Reds", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("mlp Confusion Matrix Heatmap")
plt.show()

# Overall Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Overall Accuracy:", accuracy)

# Plot Accuracy
plt.plot(l.history['accuracy'], label='Training Accuracy')
plt.plot(l.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('MLP Model Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MLP Model Loss')
plt.legend()
plt.show()

import joblib

joblib.dump(model,"mlp1.sav")









