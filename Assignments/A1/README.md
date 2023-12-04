# MNIST 数据集图像分类

## 目的

- **理解基本的图像识别流程及数据驱动的方法**：这涵盖了图像分类的整个流程，从数据准备、模型选择、训练到预测等阶段。
  
- **数据划分**：掌握如何将数据分为训练集、验证集和测试集，并理解使用验证数据如何调整模型的超参数。
  
- **Softmax分类器**：实现一个基于Softmax的图像分类器。
  
- **全连接神经网络分类器**：实现一个全连接的神经网络来分类图像。
  
- **分类器之间的区别**：理解不同分类器之间的区别，并探索使用不同的更新方法来优化神经网络。

## 附加题

- **损失函数与正则化**：尝试使用不同的损失函数和正则化方法，观察并分析其对实验结果的影响。 (+5 points)
  
- **优化算法**：尝试使用不同的优化算法，观察并分析其对训练过程和实验结果的影响。可以考虑的优化算法包括batch GD, online GD, mini-batch GD, SGD，以及其他如Momentum, Adagrad, Adam, Admax等。 (+5 points)


# 代码说明

本次实验的代码主要包括以下几个文件：
```bash
│───mnist-dense.py
│───mnist-softmax.py
│───README.md
└───utils
    │───parameters.py
    │───preprocess.py
    │───visualization.py
    │───__init__.py
```
其中，`mnist-dense.py`和`mnist-softmax.py`分别是全连接神经网络和Softmax分类器的实现代码，
`utils`文件夹中包含了一些辅助函数，如数据预处理、可视化、全局变量定义等。

本实验在可视化部分和实现softmax分类器和全连接神经网络时部分参考了来自
[tensorflow-without-a-phd ](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-mnist-tutorial)的代码。

可视化代码文件存放在`utils/visualization.py`中，我在此基础上做了一些修改。训练过程的可视化使用了`tf.keras.callbacks.Callback`类，
可以实现在训练过程中动态显示训练集和验证集的loss和accuracy。

实现softmax分类器和全连接神经网络的代码主要参考的是**网络架构定义注释部分**，即：

`mnist-softmax.py`:
```python
# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]
```

`mnist-dense.py`:
```python
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

```

其余代码和注释部分均为自己编写并实现，在代码注释部分使用了Github Copilot**辅助**生成。

# 实验内容

## 数据集简介

MNIST是一个手写数字数据集，包括了若干手写数字体及其对应的数字标签。具体数据详情如下：

- 训练样本：60000
- 测试样本：10000
- 图像尺寸：28 x 28 像素

每个手写数字被表示为一个 28*28 的像素向量。

## 模型架构

我构建了两种不同的神经网络模型：一个基于Softmax的简单分类器和一个五层的全连接神经网络。

### Softmax分类器
这是一个非常基础的模型，只有一层，结构如下：

```
X [batch, 784] → W [784, 10] + b[10] → Y [batch, 10]
```

### 全连接神经网络
这是一个复杂的模型，由五个全连接层组成，结构如下：

```
X [batch, 784]
→ W1 [784, 200] + B1[200] → Y1 [batch, 200]
→ W2 [200, 100] + B2[100] → Y2 [batch, 100]
→ W3 [100, 60] + B3[60] → Y3 [batch, 60]
→ W4 [60, 30] + B4[30] → Y4 [batch, 30]
→ W5 [30, 10] + B5[10] → Y5 [batch, 10]
```

## 实验过程

详细的实验过程，不同模型配置下的实验结果详见代码中的注释部分。

### 优化器

我首先试验了不同的优化器（optimizer），并观察了它们对模型训练的影响：

- 使用`Adam`优化器，10轮后的准确率达到了98.73%，这是我目前得到的最好结果。
- 使用纯`SGD`优化器，模型几乎没有学习，准确率仅为11.37%。但当我为`SGD`添加了0.99的动量（momentum）后，准确率提升到了97.70%。
- 使用`Adamax`和`RMSprop`优化器，得到的结果介于`Adam`和`SGD`之间。

（附加题2 +5 points）

### 损失函数

我也试验了不同的损失函数：

- 使用`categorical_crossentropy`作为损失函数，模型的准确率达到了98.73%。
- 当我改为使用`mean_squared_error`时，虽然损失下降了，但准确率也下降到了98.06%。该实验结果证明了对于分类任务，均方误差损失函数并不十分适用。

（附加题1 +5 points）
### 正则化

我尝试了Dropout作为正则化方法，并发现：

- 使用0.2的Dropout，模型的准确率下降到了95.66%。
- 当我减少Dropout率到0.1时，准确率稍微提高到了96.88%，但仍然没有达到不使用Dropout时的准确率。

这可能是因为这个模型不足够复杂，不容易出现过拟合。因此，对于结构十分简单的模型（如本模型），使用Dropout并不能提高模型的性能。

（附加题1 +5 points）
## 下一步工作

我计划进一步探索不同的模型结构，如卷积神经网络（CNN），以及其他的优化技巧，如学习率调度(learn rate scheduling)和早期停止(early stop)，
来进一步提高模型的性能和分类准确度。
