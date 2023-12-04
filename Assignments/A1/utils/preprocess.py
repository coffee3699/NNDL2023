import tensorflow as tf
from .parameters import BATCH_SIZE, EPOCHS


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

    :return: tuple\n
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
