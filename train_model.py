import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Image params
img_size = (150, 150)
batch_size = 32

# Image Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen
)

# Save model
os.makedirs("model", exist_ok=True)
model.save('model/gender_model.h5')

# Plot training metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['accuracy', 'precision', 'recall']):
        plt.subplot(1, 3, i+1)
        plt.plot(history.history[metric], label='Train')
        plt.plot(history.history['val_' + metric], label='Val')
        plt.title(metric.title())
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
    plt.tight_layout()
    plt.savefig("model/training_metrics.png")
    plt.show()

plot_metrics(history)
