import os
import tensorflow as tf

from utils.parameters import BATCH_SIZE, EPOCHS
import utils.visualization as v
from utils.preprocess import pre_process_mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
print("Tensorflow version " + tf.__version__)  # Print TF version

# Download images and labels from tf.keras.datasets.mnist API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Get training and validation datasets of tf.dataset.Dataset type
training_dataset, validation_dataset = pre_process_mnist(x_train, y_train, x_test, y_test)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(input_shape=(28 * 28,), target_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')
])

# Print model layers
model.summary()

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, min_delta=0.001)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# utility callback that displays training curves
plot_training = v.PlotTraining(sample_rate=10, zoom=5)

steps_per_epoch = 60000 // BATCH_SIZE
print("Steps per epoch: ", steps_per_epoch)

model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=validation_dataset,
          callbacks=[plot_training, early_stopping])
