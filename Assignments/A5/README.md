# 用于起名字的循环神经网络

#### 主要任务
- **数据来源**：使用CMU网站上的名字数据集，包含超过8000个英文名字。数据集地址为：[CMU名字数据集](https://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)。
- **目标**：利用这些英文名字数据训练一个循环神经网络（RNN）。当输入名字的首字母或前几个字母时，程序自动生成后续字母，直至形成完整的名字。
- **可视化技术**：采用可视化技术展示模型在每个时刻预测的前5个最可能的候选字母。

#### 附加题（10 points）
- **扩展功能**：设计并实现一个RNN模型，该模型可以基于给定的名字结尾的若干字母，或随机给出的中间字母，补全其他字母，从而形成完整的名字。
- **挑选与可视化**：从模型生成的名字中挑选一个最喜欢的名字，并使用可视化技术绘制其生成过程。

# 代码说明

本次实验的代码主要包括以下几个文件：
```bash
│───preprocess-data.py     # 数据预处理
│───Simple-RNN.py          # RNN模型的构建和训练
│───name-generator.py      # 名字生成脚本
│───README.md              # 项目说明文档
│───SimpleRNN.keras        # 训练好的模型文件
└───dataset                # 数据集文件夹
    ├───original           # 原始数据
    │   └───all_names.txt  # 所有名字的集合
    └───merged             # 合并后的数据
        ├───female.txt     # 女性名字
        ├───male.txt       # 男性名字
        ├───names.txt      # 综合名字
        └───pet.txt        # 宠物名字
```
`Simple-RNN.py`文件中，RNN模型的主要结构包括：嵌入层（Embedding），掩码层（Masking），简单循环神经网络层（SimpleRNN）以及全连接层（Dense）。
模型的训练和验证使用了CMU名字数据集，实现了基于首字母生成完整名字的功能。


`name-generator.py`文件中我们实现了一个基于循环神经网络（RNN）的英文名字生成系统，并且通过命令行界面与用户进行交互。该系统允许用户输入一个起始
字符（种子），然后生成一个完整的名字。用户可以选择接受生成的名字或请求系统再次生成。此外，用户还可以多次使用新的种子继续生成名字。

# 实验内容和过程
## 模型架构
RNN模型的架构设计如下：

```bash
def build_model(vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=10))
    model.add(Masking(mask_value=0))  # 掩蔽层，忽略填充的0
    model.add(SimpleRNN(units=100, dropout=0.15))  # RNN层，含100个单元
    model.add(Dense(vocab_size, activation='softmax'))  # 输出层，预测下一个字符
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model
```

- **Embedding层**：将字符索引映射为固定大小的稠密向量。
- **Masking层**：用于忽略序列中的填充（padding）部分。
- **SimpleRNN层**：循环神经网络层，`units=100` 表示每个时间步的输出维度是100。包含dropout正则化，减少过拟合风险。
- **Dense层**：使用softmax激活函数输出预测的下一个字符的概率分布。

## 训练过程
- 使用`train_model`函数对模型进行训练，epoch为100，batch-size为128。
  - 如果训练的epoch不够，模型生成姓名的效果会很差！
- 使用`plot_loss_curve`函数可视化训练过程中的损失下降情况。
  ![image](https://github.com/coffee3699/NNDL2023/assets/42939049/ae8f87ec-03d0-4cbc-9bc5-188d72a45e11)

## 名字生成与可视化
- `generate_name_with_visualization`函数实现了基于给定种子的名字生成过程，并通过条形图显示了模型在每一步预测的前5个最可能的候选字母。
### 举例：给定的seed为"Eli"

![image](https://github.com/coffee3699/NNDL2023/assets/42939049/e8f7ebf6-5d41-4f6d-86b8-72f37ad57e7c)
    ![image](https://github.com/coffee3699/NNDL2023/assets/42939049/917d010c-4711-4716-a1fd-17cd59687d07)
    ![image](https://github.com/coffee3699/NNDL2023/assets/42939049/fb0d6bac-b0af-4c4e-bc01-4da2257f0aba)
    ![image](https://github.com/coffee3699/NNDL2023/assets/42939049/55af7f36-11e5-4a2d-bdc9-d4177169dc52)

## 模型性能评估
我尝试了多种不同的seed输出给模型，如Ali，Eli，Bo，Trum，Step等，模型生成并输出的姓名都比较正常，说明了简单的RNN模型已经足以应付姓名生成这类简单的任务。

# 结果和讨论
本次实验中，基于RNN的名字生成模型展示了字符级文本生成的能力。实验过程中，发现模型对于不同长度的种子词的适应性以及对于生成具有一定创造性的名字的
能力。通过交叉验证和模型调优，成功提升了模型的预测准确性和生成的多样性。此外，可视化结果为模型的决策过程提供了直观的理解。
