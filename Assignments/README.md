# 神经网络与深度学习 2023 Fall

本仓库将用于存储和记录所有与该课程相关的作业。

## 运行环境配置

### 前提条件

确保您已经安装了 Anaconda。如果还没有安装，可以从 [Anaconda 官网](https://www.anaconda.com/products/distribution) 下载并安装。

### 创建新的conda环境

为了避免不同项目之间的包冲突，推荐为每个项目创建一个新的 conda 环境。

以下命令可以帮助你创建一个名为 `NNDL` 的新环境：

```bash
conda create --name NNDL python=3.11
```

### 激活conda环境

使用以下命令激活刚才创建的环境：

```bash
conda activate NNDL
```

### 安装依赖

在激活环境的状态下，切换到项目的根目录，使用以下命令从 `requirements.txt` 安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

### 退出conda环境

当完成实验后，使用下面的命令来退出当前的conda环境：

```bash
conda deactivate
```