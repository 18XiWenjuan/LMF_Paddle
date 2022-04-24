# LMF_Paddle

## 目录


- [1. 简介](#1)
- [2. 数据集和复现精度](#2)
- [3. 准备数据与环境](#3)
    - [3.1 准备环境](#3.1)
    - [3.2 准备数据](#3.2)
- [4. 开始使用](#4)
    - [4.1 模型训练](#4.1)
    - [4.2 模型评估](#4.2)
- [5. 基于Inference的推理](#5)
    - [5.1 模型导出](#5.1)
    - [5.2 模型预测](#5.2)
- [6. TIPC自动化测试脚本](#6)
- [7. LICENSE](#7)
- [8. 参考链接与文献](#8)

<a id="1"></a>
## 1. 简介

LMF是2018年Zhun Liu，Ying Shen等人提出的低秩多模态融合方法，文章主要贡献如下：

- 提出了一种低秩多模态融合方法，用于多模态融合。
- LMF与SOTA模型在公共数据集的三个多模态任务上进行了性能对比。
- 证明了LMF计算效率高，与以前的基于张量的方法相比，具有更少的参数。

![image-20220412144837152](lmf.png)

**论文:** [Efficient Low-rank Multimodal Fusion with Modality-Specific Factors](https://arxiv.org/abs/1806.00064)

**repo地址：** https://github.com/18XiWenjuan/LMF_Paddle

**项目AI Studio地址：** 请根据Notebook提示运行 https://aistudio.baidu.com/aistudio/projectdetail/3867942

<a id="2"></a>
## 2. 数据集和复现精度

数据集为IEMOCAP，它是151个录制对话的视频集合，每个会话有2个发言者，整个数据集总共有302个视频。每个片段都标注了9种情绪(愤怒、兴奋、恐惧、悲伤、惊讶、沮丧、高兴、失望和自然)。6373个视频用于训练，1775个视频用于验证，1807个视频用于测试。

[下载链接](https://pan.baidu.com/s/1rg9Pgol9MG3EZyDlmHa6bA) ，提取码：ag52

|                 | F1-Happy | F1-Sad | F1-Angry | F1-Neutral |
| --------------- | -------- | ------ | -------- | ---------- |
| LMF（论文精度） | 0.858    | 0.859  | 0.890    | 0.717      |
| LMF（复现精度） | 0.858    | 0.863  | 0.891    | 0.718      |

<a id="3"></a>
## 3. 准备环境与数据
<a id="3.1"></a>
### 3.1 准备环境

* 下载代码

```bash
git clone https://github.com/18XiWenjuan/LMF_Paddle.git
```

* 安装paddlepaddle

```bash
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.2
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.2
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```
<a id="3.2"></a>

### 3.2 准备数据

如果您已经准备IEMOCAP数据集，那么该步骤可以跳过，如果您没有，则可以从下面的链接下载，将数据集放在data文件夹中。

[下载链接](https://pan.baidu.com/s/1rg9Pgol9MG3EZyDlmHa6bA) ，提取码：ag52

如果只是希望快速体验模型训练功能，可以使用我们准备的小型数据集sample_data.pkl（位于data文件夹中，由于容量限制，sample_data.pkl只包括happy和sad两种情绪）。

<a id="4"></a>

## 4. 开始使用
<a id="4.1"></a>

### 4.1 模型训练

在scripts文件夹下运行命令：

```bash
bash train.sh
```

模型存储在 check 文件夹下，训练日志位于 train_log 文件夹下，部分训练日志如下所示：

```
Model initialized
Epoch 0 complete! Average Training loss: 0.6012711593408903
Validation loss is: 0.44860514781827615
Found new check model, saving to disk...
Epoch 1 complete! Average Training loss: 0.39506231531923114
Validation loss is: 0.42386036349418466
Found new check model, saving to disk...
Epoch 2 complete! Average Training loss: 0.37492612341023296
Validation loss is: 0.4046861079701206
```
<a id="4.2"></a>

### 4.2 模型评估

在scripts文件夹下运行命令：

```bash
bash eval.sh
```

输出示例：

```
Start testing happy......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
Model initialized
F1-score on test set is 0.8583335303191512
Accuracy score on test set is 0.8731343283582089

Start testing sad......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
Model initialized
F1-score on test set is 0.862597215789817
Accuracy score on test set is 0.8667377398720683

Start testing angry......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
Model initialized
F1-score on test set is 0.8912247645604259
Accuracy score on test set is 0.8923240938166311

Start testing neutral......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
Model initialized
F1-score on test set is 0.7183055564744968
Accuracy score on test set is 0.7217484008528785
```

<a id="5"></a>

## 5. 基于Inference的推理
<a id="5.1"></a>

### 5.1 模型导出

export.py 用于将训练模型动转静导出，在scripts文件夹下运行命令：

```bash
bash export.sh
```

静态模型会保存到 check_inference 文件夹中：

```
check_inference
     |----angry.pdiparams     : 模型参数文件
     |----angry.pdiparams.info: 模型参数信息文件
     |----angry.pdmodel       : 模型结构文件
     |----happy.pdiparams
     |----happy.pdiparams.info
     |----happy.pdmodel
     |----neutral.pdiparams
     |----neutral.pdiparams.info
     |----neutral.pdmodel
     |----sad.pdiparams
     |----sad.pdiparams.info
     |----sad.pdmodel
```
<a id="5.2"></a>

### 5.2 模型预测

infer.py 用于模型在未知数据上的推理测试，在scripts文件夹下运行命令：

```bash
bash infer.sh
```

在当前分类任务中将预测概率结果保存到 out_prob 文件夹中：

```
out_prob
     |----angry_prob.npy
     |----happy_prob.npy
     |----neutral_prob.npy
     |----sad_prob.npy
```
<a id="6"></a>

## 6. TIPC自动化测试脚本

以Linux基础训练推理测试为例，我们准备了小型数据集sample_data.pkl（位于data文件夹中，由于容量限制，sample_data.pkl只包括happy和sad两种情绪，也可以在原始的iemocap数据集上运行）用来测试TIPC自动化脚本，具体见[README.md](/test_tipc/README.md)。

测试命令如下：

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/lmf/train_infer_python.txt lite_train_lite_infer
```

具体请参照[test_train_inference_python.md](./test_tipc/docs/test_train_inference_python.md)

<a id="7"></a>

## 7. LICENSE

本项目的发布受[MIT license](https://github.com/simonsLiang/PReNet_paddle/blob/main/LICENSE)许可认证。

<a id="8"></a>

## 8. 参考链接与文献

1. [Efficient Low-rank Multimodal Fusion with Modality-Specific Factors](https://arxiv.org/abs/1806.00064)
2. 参考repo：[Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)