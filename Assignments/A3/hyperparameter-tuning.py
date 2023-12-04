import tensorflow as tf

from scikeras.wrappers import KerasClassifier
from utils.parameters import BATCH_SIZE, EPOCHS
from sklearn.model_selection import GridSearchCV

# Load the MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the dataset to be in the range [0, 1]
x_train = x_train.astype('float32') / 255.0

# Flatten the dataset from (28, 28) to (784,)
x_train = x_train.reshape(-1, 28 * 28)

# Convert labels to one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, 10)


# Wrap the model architecture in a function
def create_model(optimizer='adam', dropout_rate=0.25):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(28 * 28,), target_shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        tf.keras.layers.Conv2D(filters=12, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=200, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Wrap the model function in KerasClassifier
mnist_classifier = KerasClassifier(
    model=create_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    dropout_rate=None
)

# Define hyperparameters and their potential values
param_grid = {
    'optimizer': ['SGD', 'Adam', 'RMSprop'],
    'dropout_rate': [0.2, 0.25, 0.3],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20]
}

# Apply GridSearchCV
grid = GridSearchCV(estimator=mnist_classifier, param_grid=param_grid, cv=3, verbose=1)
grid_result = grid.fit(x_train, y_train)

# Print the best results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

"""
Some of the log results:

Fitting 3 folds for each of 54 candidates, totalling 162 fits
[CV 1/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=SGD;, score=0.984 total time= 1.3min
[CV 2/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=SGD;, score=0.983 total time= 1.5min
[CV 3/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=SGD;, score=0.986 total time= 1.4min
[CV 1/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=Adam;, score=0.987 total time= 1.4min
[CV 2/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=Adam;, score=0.984 total time= 1.1min
[CV 3/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=Adam;, score=0.984 total time= 1.5min
[CV 1/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=RMSprop;, score=0.982 total time= 1.4min
[CV 2/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=RMSprop;, score=0.985 total time= 1.1min
[CV 3/3] END batch_size=32, dropout_rate=0.2, epochs=10, optimizer=RMSprop;, score=0.983 total time= 1.4min
[CV 1/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=SGD;, score=0.989 total time= 2.4min
[CV 2/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=SGD;, score=0.987 total time= 2.1min
[CV 3/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=SGD;, score=0.987 total time= 2.1min
[CV 1/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=Adam;, score=0.990 total time= 2.4min
[CV 2/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=Adam;, score=0.986 total time= 2.4min
[CV 3/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=Adam;, score=0.988 total time= 2.5min
[CV 1/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=RMSprop;, score=0.987 total time= 2.4min
[CV 2/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=RMSprop;, score=0.984 total time= 2.4min
[CV 3/3] END batch_size=32, dropout_rate=0.2, epochs=20, optimizer=RMSprop;, score=0.988 total time= 2.4min
[CV 1/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=SGD;, score=0.986 total time= 1.1min
[CV 2/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=SGD;, score=0.983 total time= 1.4min
[CV 3/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=SGD;, score=0.984 total time= 1.1min
[CV 1/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=Adam;, score=0.983 total time= 1.1min
[CV 2/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=Adam;, score=0.984 total time= 1.1min
[CV 3/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=Adam;, score=0.981 total time= 1.4min
[CV 1/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=RMSprop;, score=0.985 total time= 1.5min
[CV 2/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=RMSprop;, score=0.986 total time= 1.4min
[CV 3/3] END batch_size=32, dropout_rate=0.25, epochs=10, optimizer=RMSprop;, score=0.982 total time= 1.4min
[CV 1/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=SGD;, score=0.988 total time= 2.4min
[CV 2/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=SGD;, score=0.984 total time= 2.4min
[CV 3/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=SGD;, score=0.986 total time= 2.1min
[CV 1/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=Adam;, score=0.987 total time= 2.4min
[CV 2/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=Adam;, score=0.988 total time= 2.1min
[CV 3/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=Adam;, score=0.988 total time= 2.4min
[CV 1/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=RMSprop;, score=0.987 total time= 2.5min
[CV 2/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=RMSprop;, score=0.985 total time= 2.1min
[CV 3/3] END batch_size=32, dropout_rate=0.25, epochs=20, optimizer=RMSprop;, score=0.987 total time= 2.1min
[CV 1/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=SGD;, score=0.986 total time= 1.4min
[CV 2/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=SGD;, score=0.979 total time= 1.1min
[CV 3/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=SGD;, score=0.985 total time= 1.5min
[CV 1/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=Adam;, score=0.981 total time= 1.1min
[CV 2/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=Adam;, score=0.978 total time= 1.4min
[CV 3/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=Adam;, score=0.974 total time= 1.1min
[CV 1/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=RMSprop;, score=0.986 total time= 1.4min
[CV 2/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=RMSprop;, score=0.980 total time= 1.1min
[CV 3/3] END batch_size=32, dropout_rate=0.3, epochs=10, optimizer=RMSprop;, score=0.983 total time= 1.1min
[CV 1/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=SGD;, score=0.987 total time= 2.1min
[CV 2/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=SGD;, score=0.984 total time= 2.1min
[CV 3/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=SGD;, score=0.986 total time= 2.4min
[CV 1/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=Adam;, score=0.987 total time= 2.1min
[CV 2/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=Adam;, score=0.985 total time= 2.1min
[CV 3/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=Adam;, score=0.985 total time= 2.4min
[CV 1/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=RMSprop;, score=0.986 total time= 2.2min
[CV 2/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=RMSprop;, score=0.981 total time= 2.5min
[CV 3/3] END batch_size=32, dropout_rate=0.3, epochs=20, optimizer=RMSprop;, score=0.985 total time= 2.4min
[CV 1/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=SGD;, score=0.985 total time=  35.3s
[CV 2/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=SGD;, score=0.984 total time=  35.7s
[CV 3/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=SGD;, score=0.982 total time=  44.4s
[CV 1/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=Adam;, score=0.986 total time=  35.1s
[CV 2/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=Adam;, score=0.984 total time=  45.0s
[CV 3/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=Adam;, score=0.983 total time=  44.3s
[CV 1/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=RMSprop;, score=0.985 total time=  44.4s
[CV 2/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=RMSprop;, score=0.985 total time=  44.0s
[CV 3/3] END batch_size=64, dropout_rate=0.2, epochs=10, optimizer=RMSprop;, score=0.984 total time=  34.9s
[CV 1/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=SGD;, score=0.989 total time= 1.4min
[CV 2/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=SGD;, score=0.987 total time= 1.1min
[CV 3/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=SGD;, score=0.985 total time= 1.1min
[CV 1/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=Adam;, score=0.989 total time= 1.1min
[CV 2/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=Adam;, score=0.986 total time= 1.1min
[CV 3/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=Adam;, score=0.985 total time= 1.1min
[CV 1/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=RMSprop;, score=0.986 total time= 1.1min
[CV 2/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=RMSprop;, score=0.988 total time= 1.4min
[CV 3/3] END batch_size=64, dropout_rate=0.2, epochs=20, optimizer=RMSprop;, score=0.987 total time= 1.1min
[CV 1/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=SGD;, score=0.987 total time=  44.4s
[CV 2/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=SGD;, score=0.983 total time=  35.6s
[CV 3/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=SGD;, score=0.983 total time=  44.4s
[CV 1/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=Adam;, score=0.985 total time=  35.1s
[CV 2/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=Adam;, score=0.984 total time=  35.7s
[CV 3/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=Adam;, score=0.984 total time=  44.4s
[CV 1/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=RMSprop;, score=0.985 total time=  43.8s
[CV 2/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=RMSprop;, score=0.983 total time=  34.9s
[CV 3/3] END batch_size=64, dropout_rate=0.25, epochs=10, optimizer=RMSprop;, score=0.984 total time=  43.9s
"""
