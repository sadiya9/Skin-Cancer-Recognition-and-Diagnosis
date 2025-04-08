
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


train=ImageDataGenerator(rescale=1/255)
train_data=train.flow_from_directory("skin cancer\\train",
                                    batch_size=3,
                                    target_size=(250,250))
validation_data=train.flow_from_directory("skin cancer\\test",
                                    batch_size=3,
                                    target_size=(250,250))




base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(250, 250, 3))


for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

sol=model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20
)

print("Training Accuracy:", max(sol.history["accuracy"]))
print("Validation Accuracy:", max(sol.history["val_accuracy"]))

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
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Overall Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Overall Accuracy:", accuracy)



# Plot Accuracy
plt.plot(sol.history['accuracy'], label='Training Accuracy')
plt.plot(sol.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ResNet Model Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(sol.history['loss'], label='Training Loss')
plt.plot(sol.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ResNet Model Loss')
plt.legend()
plt.show()

import joblib

joblib.dump(model,"resnet.sav")



