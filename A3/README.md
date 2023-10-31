# 基于CNN的图像分类

## 目的

- **理解卷积神经网络的基本结构、代码实现及训练过程**：卷积神经网络是目前图像识别领域最流行的模型。你将通过本实验了解CNN的基本构成、其代码实现方式，以及训练CNN的基本过程。

- **应用dropout和多种normalization方法**：dropout和normalization是神经网络训练中常用的正则化方法，可以有效避免过拟合，提高模型的泛化能力。在这部分，你将了解如何在CNN中应用这些方法，并理解它们对模型泛化能力的影响。

- **交叉验证寻找最优超参数**：超参数的选择对模型的性能有很大的影响。在这部分，你将使用交叉验证的方法，为你的CNN找到一组最优的超参数(hyperparameters)。

## 数据集

- **MNIST 或 CIFAR-10**：MNIST是一个手写数字识别的数据集，而CIFAR-10包含10个类别的彩色图片。你可以选择其中一个，或者都使用来完成本实验。

## 附加题 (+5 points)

- **探索性能提升方法**：在训练网络的过程中，你可以自由尝试其他提升性能的方法。以下是一些建议的探索方向：
  
  - **模型深度**：尝试增加模型的层数，看看是否可以进一步提高性能。
  
  - **正则化方法**：除了dropout和normalization，还有其他的正则化方法，如L1/L2正则化、早停 (early stopping) 等，可以尝试结合使用看其对模型性能的影响。
  
  - **模型集成**：通过结合多个模型的预测来提高整体的预测准确率。
  
对于以上探索所得到的性能提升，你可以获得额外的+5 points。

# 代码说明

本次实验的代码主要包括以下几个文件：
```bash
│───hyperparameter-tuning.py
│───mnist-convolutional.py
│───mnist-betterCNN.py
│───README.md
└───utils
    │───parameters.py
    │───preprocess.py
    │───visualization.py
    │───__init__.py
```

其中，`mnist-convolutional.py`是卷积神经网络简易版本的实现，在经过交叉验证的超参数优化后最终的准确率为99.0%。
而`hyperparameter-tuning.py`是使用交叉验证进行超参数优化的代码，最终得到的最优超参数为`batch_size=32, dropout_rate=0.2, epochs=20, optimizer=Adam`。

`mnist-betterCNN.py`是卷积神经网络改进版本的实现，最终的验证集准确率为99.6%，与简易版本相比有了很大的提升。

`utils`文件夹中包含了一些辅助函数，如数据预处理、可视化、全局变量定义等。

> 代码和注释部分均为自己编写并实现，在代码注释部分使用了Github Copilot**辅助**生成。


# 实验内容

## 模型架构

### 简单CNN模型

在一开始，我构建的简单CNN模型架构为：

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(filters=12, kernel_size=3, strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=200, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
```

在该模型中，我使用了三个卷积层，每个卷积层后面都跟着一个ReLU激活函数。在第一个卷积层后面，我加入了一个最大池化层，以减少模型的参数数量。
在最后的全连接层中，我使用了ReLU激活函数。

在实验中，由于模型参数复杂或未使用正则化等原因，导致模型产生了过拟合现象。从训练结果可以看出，训练准确率为0.9954，但验证准确率仅为0.9899。

不过，相比于第一次实验构建的全连接神经网络和简单softmax分类器，这个简单的CNN模型已经取得了很大的进步。

### CNN+正则化方法

接下来，我在现有模型的基础上加入了一些正则化方法，包括dropout和batch normalization来尝试改善模型在训练数据上过拟合的情况。

模型架构如下：

```
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
```

在该模型中，我在每个卷积层和全连接层后面都加入了batch normalization层，以及dropout层。

从结果上来看，在加入了正则化方法后，模型的过拟合现象得到了一定程度的缓解，最终的训练准确率为0.9715，而验证准确率为0.9819。

但是模型的性能在加入正则化方法后出现了小幅度的下降，因此需要我们再尝试更优的正则化方法，以及对模型的网络结构方面进行改进。

## 使用交叉验证进行超参数调优

### 概念梳理

- **超参数**：超参数是指在训练神经网络时，需要人为设定的参数。主要包含有：epoch、学习率、batch size、网络结构、正则化方法等。
- **交叉验证**：交叉验证是一种模型评估的方法，它将数据集分为训练集和验证集，然后使用训练集训练模型，使用验证集评估模型的性能。 
在交叉验证中，我们可以通过多次划分训练集和验证集，来评估模型的性能。最常用的交叉验证方法是k折交叉验证，即将训练集分为k份，每次使用其中一份作为验证集，其余k-1份作为训练集，然后重复k次，最后将k次的评估结果取平均值作为模型的最终评估结果。

### 实验过程

在本次实验中，我通过使用`scikeras`库中的`KerasClassifier`类，将`tf.keras`模型转换为`scikit-learn`模型，从而可以使用`scikit-learn`库中的使用交叉验证进行超参数调优的方法。

进一步地，我定义了一些可选的超参数（为了节省计算资源，我没有定义很多超参数），包括：

- 优化器：SGD、Adam、RMSprop
- Dropout率：0.2、0.25、0.3
- Batch size：32、64、128
- Epochs：10、20

使用`GridSearchCV`方法，对这些超参数进行了组合，最终得到了一组最优的超参数：

> 在`GridSearchCV`中，参数`cv`即cross validation，表示使用k折交叉验证，这里我使用了k=3，即将训练集分为3份，每次使用其中一份作为验证集，其余两份作为训练集，然后重复3次，最后将3次的评估结果取平均值作为模型的最终评估结果。

- batch_size=32, dropout_rate=0.2, epochs=20, optimizer=Adam

> 在Google Colab上，使用'GridSearchCV'方法进行超参数调优的过程大约需要2小时左右（very expensive!!!）。

## 附加题 (+5 points)

在使用交叉验证进行模型的超参数优化后，我们的分类准确率已经来到了99.0%。

不过，想要突破这个瓶颈，我们从增加模型层数和增加更多的正则化方法（如早停）这两方面入手，尝试进一步提高模型的性能。

经过多次实验结果验证以及互联网上的SOTA经验分享，我们重新定义的模型架构如下：

```bash
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(input_shape=(28*28,), target_shape=(28, 28, 1)),
  
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', input_shape=(28,28,1)),
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
```
在经过20个epoch的训练后，模型的训练准确率达到了99.7%，验证准确率达到了99.5%，相比于之前的模型，在避免了过拟合的同时模型性能有了很大的提升。

这证明了增加模型层数（深度）和使用适当的正则化方法对于提升模型性能很有帮助，但是这一切的前提是我们需要首先对于模型的基础架构有一个清晰的认识，否则消耗再多的计算资源也无法改变模型的性能。

