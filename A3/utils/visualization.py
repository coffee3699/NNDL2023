"""
visualization utilities

This module contains helper functions used for visualization
and downloads only.

You can skip reading it. There is very
little useful Keras/Tensorflow code here.

Credit: @martin_gorner
"""
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Define some global variables
BATCH_SIZE = 128
EPOCHS = 10

# Matplotlib config
plt.ioff()
plt.rc('image', cmap='gray_r')
plt.rc('grid', linewidth=1)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0', figsize=(16, 9))
# Matplotlib fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")


class PlotTraining(tf.keras.callbacks.Callback):
    """
    Used to plot training progress in local environment.

    Original code is specifically tailored for Jupyter notebook or IPython environment.
    I (Dongyuan Li) have modified it to work with local environment in PyCharm.
    """
    def __init__(self, sample_rate=1, zoom=1):
        super().__init__()
        self.sample_rate = sample_rate
        self.step = 0
        self.zoom = zoom
        self.steps_per_epoch = 60000 // BATCH_SIZE  # Adjust this if BATCH_SIZE is not a global variable

    def on_train_begin(self, logs={}):
        self.batch_history = {}
        self.batch_step = []
        self.epoch_history = {}
        self.epoch_step = []
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))
        plt.ioff()

    def on_batch_end(self, batch, logs={}):
        if (batch % self.sample_rate) == 0:
            self.batch_step.append(self.step)
            for k, v in logs.items():
                if k == 'batch' or k == 'size':
                    continue
                self.batch_history.setdefault(k, []).append(v)
        self.step += 1

    def on_epoch_end(self, epoch, logs={}):
        plt.close(self.fig)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))

        self.axes[0].set_ylim(0, 1.2 / self.zoom)
        self.axes[1].set_ylim(1 - 1 / self.zoom / 2, 1 + 0.1 / self.zoom / 2)

        self.epoch_step.append(self.step)
        for k, v in logs.items():
            if not k.startswith('val_'):
                continue
            self.epoch_history.setdefault(k, []).append(v)

        for k, v in self.batch_history.items():
            self.axes[0 if k.endswith('loss') else 1].plot(np.array(self.batch_step) / self.steps_per_epoch, v, label=k)

        for k, v in self.epoch_history.items():
            self.axes[0 if k.endswith('loss') else 1].plot(np.array(self.epoch_step) / self.steps_per_epoch, v, label=k,
                                                           linewidth=3)

        self.axes[0].legend()
        self.axes[1].legend()
        self.axes[0].set_xlabel('epochs')
        self.axes[1].set_xlabel('epochs')
        self.axes[0].minorticks_on()
        self.axes[0].grid(True, which='major', axis='both', linestyle='-', linewidth=1)
        self.axes[0].grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)
        self.axes[1].minorticks_on()
        self.axes[1].grid(True, which='major', axis='both', linestyle='-', linewidth=1)
        self.axes[1].grid(True, which='minor', axis='both', linestyle=':', linewidth=0.5)

        plt.show(block=False)
        plt.pause(0.1)  # Adjust this delay as needed
