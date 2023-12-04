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
    tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Add batch normalization
    tf.keras.layers.Dropout(0.25),  # Add dropout with rate 0.25

    tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Add batch normalization
    tf.keras.layers.Dropout(0.25),  # Add dropout with rate 0.25

    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

    tf.keras.layers.Conv2D(filters=12, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Add batch normalization
    tf.keras.layers.Dropout(0.25),  # Add dropout with rate 0.25

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=200, activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Add batch normalization
    tf.keras.layers.Dropout(0.5),  # Add dropout with rate 0.5 for fully connected layers

    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Print model layers
model.summary()

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# utility callback that displays training curves
plot_training = v.PlotTraining(sample_rate=10, zoom=5)

steps_per_epoch = 60000 // BATCH_SIZE
print("Steps per epoch: ", steps_per_epoch)

model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=validation_dataset,
          callbacks=[plot_training])

"""
Results:

1st model architecture (simple CNN):

Epoch 1/10
600/600 [==============================] - 15s 16ms/step - loss: 0.2839 - accuracy: 0.9158 - val_loss: 0.0899 - val_accuracy: 0.9715
Epoch 2/10
600/600 [==============================] - 11s 18ms/step - loss: 0.0797 - accuracy: 0.9753 - val_loss: 0.0570 - val_accuracy: 0.9804
Epoch 3/10
600/600 [==============================] - 11s 19ms/step - loss: 0.0551 - accuracy: 0.9827 - val_loss: 0.0537 - val_accuracy: 0.9817
Epoch 4/10
600/600 [==============================] - 12s 21ms/step - loss: 0.0449 - accuracy: 0.9854 - val_loss: 0.0486 - val_accuracy: 0.9851
Epoch 5/10
600/600 [==============================] - 12s 20ms/step - loss: 0.0323 - accuracy: 0.9904 - val_loss: 0.0426 - val_accuracy: 0.9856
Epoch 6/10
600/600 [==============================] - 14s 23ms/step - loss: 0.0306 - accuracy: 0.9904 - val_loss: 0.0401 - val_accuracy: 0.9876
Epoch 7/10
600/600 [==============================] - 14s 23ms/step - loss: 0.0252 - accuracy: 0.9916 - val_loss: 0.0395 - val_accuracy: 0.9881
Epoch 8/10
600/600 [==============================] - 14s 23ms/step - loss: 0.0210 - accuracy: 0.9933 - val_loss: 0.0343 - val_accuracy: 0.9893
Epoch 9/10
600/600 [==============================] - 14s 23ms/step - loss: 0.0189 - accuracy: 0.9938 - val_loss: 0.0393 - val_accuracy: 0.9884
Epoch 10/10
600/600 [==============================] - 14s 23ms/step - loss: 0.0140 - accuracy: 0.9954 - val_loss: 0.0368 - val_accuracy: 0.9899


From the results, we can see that the model is overfitting the training data. The training accuracy is 0.9954, but the
validation accuracy is only 0.9899. This means that the model is not generalizing well to unseen data.

2nd model architecture (CNN with batch normalization and dropout):

Epoch 1/10
600/600 [==============================] - 22s 29ms/step - loss: 0.5425 - accuracy: 0.8327 - val_loss: 0.2166 - val_accuracy: 0.9318
Epoch 2/10
600/600 [==============================] - 18s 30ms/step - loss: 0.2145 - accuracy: 0.9317 - val_loss: 0.1245 - val_accuracy: 0.9593
Epoch 3/10
600/600 [==============================] - 20s 34ms/step - loss: 0.1639 - accuracy: 0.9485 - val_loss: 0.1059 - val_accuracy: 0.9642
Epoch 4/10
600/600 [==============================] - 22s 37ms/step - loss: 0.1389 - accuracy: 0.9563 - val_loss: 0.0800 - val_accuracy: 0.9728
Epoch 5/10
600/600 [==============================] - 22s 37ms/step - loss: 0.1261 - accuracy: 0.9603 - val_loss: 0.0812 - val_accuracy: 0.9719
Epoch 6/10
600/600 [==============================] - 26s 43ms/step - loss: 0.1198 - accuracy: 0.9616 - val_loss: 0.0730 - val_accuracy: 0.9756
Epoch 7/10
600/600 [==============================] - 23s 38ms/step - loss: 0.1096 - accuracy: 0.9652 - val_loss: 0.0545 - val_accuracy: 0.9803
Epoch 8/10
600/600 [==============================] - 23s 38ms/step - loss: 0.1007 - accuracy: 0.9679 - val_loss: 0.0507 - val_accuracy: 0.9828
Epoch 9/10
600/600 [==============================] - 24s 40ms/step - loss: 0.0941 - accuracy: 0.9699 - val_loss: 0.0485 - val_accuracy: 0.9840
Epoch 10/10
600/600 [==============================] - 24s 39ms/step - loss: 0.0889 - accuracy: 0.9715 - val_loss: 0.0521 - val_accuracy: 0.9819

From the results, we can see that after implementing batch normalization and dropout, the overfitting problem is resolved.
The training accuracy is 0.9715, and the validation accuracy is 0.9819. The validation accuracy is even higher than the
training accuracy, which means that the model is generalizing well to unseen data.
"""

