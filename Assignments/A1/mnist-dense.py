import os
import tensorflow as tf

from utils.parameters import BATCH_SIZE, EPOCHS
import utils.visualization as v
from utils.preprocess import pre_process_mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
print("Tensorflow version " + tf.__version__)  # Print TF version


# neural network with 5 layers
#
# · · · · · · · · · ·          (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/       -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                                Y1 [batch, 200]
#   \x/x\x/x\x/x\x/         -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                                  Y2 [batch, 100]
#     \x/x\x/x\x/           -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#      · · · · ·                                                    Y3 [batch, 60]
#       \x/x\x/             -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#        · · ·                                                      Y4 [batch, 30]
#         \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#          ·                                                        Y5 [batch, 10]

"""
Some results from previous experiments:

## Best result so far:
## optimizer = 'Adam', loss = 'categorical_crossentropy', 10 epochs
#  loss: 0.0469 - accuracy: 0.9873 - val_loss: 0.0258 - val_accuracy: 0.9900

If we change the activation function to 'relu' for all layers (except the last one), we get:
loss: 0.0019 - accuracy: 0.9884 - val_loss: 0.0022 - val_accuracy: 0.9800
which is slightly better than with sigmoid.

This improvement applies to all the following experiments.

------------------------------------------------------------------------------------
## Experiment with different optimizers:

## optimizer = 'Adam', loss = 'categorical_crossentropy', 10 epochs
#  loss: 0.0469 - accuracy: 0.9873 - val_loss: 0.0258 - val_accuracy: 0.9900

## optimizer = 'sgd', loss = 'categorical_crossentropy', 20 epochs
#  loss: 2.2982 - accuracy: 0.1137 - val_loss: 2.2935 - val_accuracy: 0.1400
#  Note: loss is too high, accuracy is too low!!! (model is basically not learning)
#        Pure SGD is bad for this (multi-layer) FC model.
#
#        In keras, SGD is implemented with momentum=0.0 by default.
#        If we set momentum=0.99, we get:
#        loss: 0.0771 - accuracy: 0.9770 - val_loss: 0.0412 - val_accuracy: 0.9900
#        which is much better, but still not as good as Adam.

## optimizer = 'Adamax', loss = 'categorical_crossentropy', 10 epochs
#  loss: 0.1769 - accuracy: 0.9509 - val_loss: 0.0832 - val_accuracy: 0.9700

## optimizer = 'RMSprop', loss = 'categorical_crossentropy', 10 epochs
#  loss: 0.0865 - accuracy: 0.9753 - val_loss: 0.0636 - val_accuracy: 0.9800

------------------------------------------------------------------------------------
## Experiment with different loss functions:

## optimizer = 'Adam', loss = 'categorical_crossentropy', 10 epochs
#  loss: 0.0469 - accuracy: 0.9873 - val_loss: 0.0258 - val_accuracy: 0.9900

## optimizer = 'Adam', loss = 'mean_squared_error', 10 epochs
#  loss: 0.0032 - accuracy: 0.9806 - val_loss: 0.0027 - val_accuracy: 0.9700
#  Note: Using MSE, we observe that the loss is much lower than with cross-entropy, but the accuracy is also lower and 
#        not improving much with more iterations. 
#        This is because MSE is not a good loss function for classification tasks.

------------------------------------------------------------------------------------
## Experiment with different normalization techniques:

## optimizer = 'Adam', loss = 'categorical_crossentropy', 10 epochs, no normalization/regularization
#  loss: 0.0469 - accuracy: 0.9873 - val_loss: 0.0258 - val_accuracy: 0.9900

## optimizer = 'Adam', loss = 'categorical_crossentropy', 10 epochs, with dropout=0.2
#  loss: 0.0072 - accuracy: 0.9566 - val_loss: 0.0021 - val_accuracy: 0.9900
#  Note: Using dropout, we observe that the loss is much lower than without dropout, but the accuracy is also lower and
#        not improving much with more iterations.
#
#        This is because dropout is hindering the model's learning. In this case,the dropout rate is too high, it can
#        be too aggressive, effectively preventing the model from learning the underlying patterns.
#       
#        If we set dropout=0.1, we get:
#        loss: 0.0051 - accuracy: 0.9688 - val_loss: 8.8250e-04 - val_accuracy: 0.9900
#        which is slightly better, but still not as good as without dropout.
#
#        I think the reason behind is that this model is not complex enough to require dropout. Dropout is useful when 
#        model is complex and overfits the training data. In this case, the model is not complex enough to overfit the
#        training data, so dropout is not needed.
"""

# Download images and labels from tf.keras.datasets.mnist API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Get training and validation datasets of tf.dataset.Dataset type
training_dataset, validation_dataset = pre_process_mnist(x_train, y_train, x_test, y_test)

# Define the model architecture
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28 * 28,)),
        tf.keras.layers.Dense(200, activation='sigmoid'),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(100, activation='sigmoid'),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(60, activation='sigmoid'),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(30, activation='sigmoid'),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Define the model optimizer, loss function and metrics (a.k.a compile the model in keras)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mean_squared_error',
              metrics=['accuracy'])

# print model layers (for inspection)
model.summary()

# utility callback that displays training curves
plot_training = v.PlotTraining(sample_rate=10, zoom=1)

steps_per_epoch = 60000 // BATCH_SIZE
print("Steps per epoch: ", steps_per_epoch)

history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_dataset, validation_steps=1, callbacks=[plot_training])
