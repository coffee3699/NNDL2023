import os
import tensorflow as tf

from utils.parameters import BATCH_SIZE, EPOCHS
import utils.visualization as v
from utils.preprocess import pre_process_mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
print("Tensorflow version " + tf.__version__)   # Print TF version

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]


# Download images and labels from tf.keras.datasets.mnist API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Get training and validation datasets
training_dataset, validation_dataset = pre_process_mnist(x_train, y_train, x_test, y_test)

# Define the model architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28 * 28,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Define the model optimizer, loss function and metrics (a.k.a compile the model in keras)
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print model layers (for inspection)
model.summary()

# utility callback that displays training curves
plot_training = v.PlotTraining(sample_rate=10, zoom=1)

steps_per_epoch = 60000 // BATCH_SIZE
print("Steps per epoch: ", steps_per_epoch)

history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_dataset, validation_steps=1, callbacks=[plot_training])
