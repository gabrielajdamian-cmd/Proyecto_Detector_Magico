import tensorflow as tf
from keras.applications import MobileNetV2
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# 1. Cargar datos con aumento de imágenes —  CORREGIDO para MobileNetV2
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,  #  Reemplaza rescale
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    subset='training'
)


val_data = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


# 2. Transfer Learning con MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


# 3. Construir la arquitectura del modelo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 4. Entrenar el modelo (10 épocas)
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)


# 5. Guardar el modelo en formato .h5
model.save('detector_magico_model.h5')
print("\n¡Modelo guardado exitosamente como 'detector_magico_model.h5'!")


# 6. Reporte de Clasificación y Gráficas
val_data.reset()
predictions = model.predict(val_data)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes


print("\n--- Reporte de Clasificación ---")
etiquetas = list(train_data.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=etiquetas))


# Graficar precisión y pérdida
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Accuracy del Modelo')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida (Loss) del Modelo')
plt.legend()
plt.tight_layout()
plt.savefig('grafica_rendimiento.png')
plt.show()


# 7. Graficar y guardar la Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=etiquetas,
    yticklabels=etiquetas,
)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('matriz_confusion.png')
plt.show()
