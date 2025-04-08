
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.optimizers import AdamW
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


train=ImageDataGenerator(rescale=1/255,
    rotation_range=30,  
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2,  # Shift images vertically
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,  
    fill_mode="nearest")
validation=ImageDataGenerator(rescale=1/255)
train_data=train.flow_from_directory("skin cells\\Train",
                                    batch_size=8,
                                    target_size=(250,250)
                                    )
validation_data=train.flow_from_directory("skin cells\\Test",
                                    batch_size=8,
                                    target_size=(250,250))




base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(250, 250, 3))


for layer in base_model.layers:
    layer.trainable = True


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(9, activation="softmax")(x)

# Final Model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the Model
model.compile(optimizer=AdamW(learning_rate=1e-5,weight_decay=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the Model
ans=model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20
)
print("Training Accuracy:", max(ans.history["accuracy"]))
print("Validation Accuracy:", max(ans.history["val_accuracy"]))

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
sns.heatmap(cm, annot=True, cmap="Purples", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Resnet Confusion Matrix Heatmap")
plt.show()

# Overall Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Overall Accuracy:", accuracy)

# Plot Accuracy
plt.plot(ans.history['accuracy'], label='Training Accuracy')
plt.plot(ans.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('ResNet1 Model Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(ans.history['loss'], label='Training Loss')
plt.plot(ans.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ResNet1 Model Loss')
plt.legend()
plt.show()


import joblib

joblib.dump(model,"resnet1.sav")



