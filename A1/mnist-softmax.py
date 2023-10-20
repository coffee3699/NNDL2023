import os
import tensorflow as tf

from utils.parameters import BATCH_SIZE, EPOCHS
import utils.visualization as v

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
print("Tensorflow version " + tf.__version__)   # Print TF version

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

def pre_process_mnist(x_train, y_train, x_test, y_test):
    """
    Pre-process the MNIST dataset from keras dataset API for future training and validation.

    The function performs the following operations:

    - 1. Flatten the input images.\n
    - 2. Normalize the pixel values of the images to the range [0, 1].\n
    - 3. One-hot encode the labels.\n
    - 4. Create tf.data datasets for training and validation.\n

    `num_samples` should be 60000 in MNIST training dataset and 10000 in MNIST test dataset.

    :param x_train: np.ndarray
        Training data containing raw images. Expected shape is (num_samples, 28, 28).
    :param y_train: np.ndarray
        Training labels corresponding to x_train. Expected shape is (num_samples,).
    :param x_test: np.ndarray
        Test data containing raw images. Expected shape is (num_samples, 28, 28).
    :param y_test: np.ndarray
        Test labels corresponding to x_test. Expected shape is (num_samples,).

    :return: tuple
        A tuple containing two elements:
        - training_dataset: tf.data.Dataset
            Dataset object ready for training. Batches and shuffles data.
        - validation_dataset: tf.data.Dataset
            Dataset object ready for validation/testing. Batches data.
    """
    # Flatten and normalize the dataset
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # If .repeat() is not used, the dataset will be exhausted after one iteration (one epoch).
    # In terminal, you will see warnings like:
    #
    # WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset
    # or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 4680 batches).
    # You may need to use the repeat() function when building your dataset.
    #
    training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat(EPOCHS).batch(BATCH_SIZE).shuffle(
        buffer_size=10000)
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).repeat(EPOCHS).batch(BATCH_SIZE)

    return training_dataset, validation_dataset


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

steps_per_epoch = 60000 // BATCH_SIZE  # 60,000 items in this dataset
print("Steps per epoch: ", steps_per_epoch)

history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_dataset, validation_steps=1, callbacks=[plot_training])