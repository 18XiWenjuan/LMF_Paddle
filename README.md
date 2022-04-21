
# LMF_Paddle

基于Paddle复现《Efficient Low-rank Multimodal Fusion with Modality-Specific Factors》

## 1. 简介

LMF是2018年Zhun Liu，Ying Shen等人提出的低秩多模态融合方法，文章主要贡献如下：

- 提出了一种低秩多模态融合方法，用于多模态融合。
- LMF与SOTA模型在公共数据集的三个多模态任务上进行了性能对比。
- 证明了LMF计算效率高，与以前的基于张量的方法相比，具有更少的参数。

![image-20220412144837152](lmf.png)


## 2. 数据集和复现精度

### 数据集：IEMOCAP


IEMOCAP数据集是151个录制对话的视频集合，每个会话有2个发言者，整个数据集总共有302个视频。每个片段都标注了9种情绪(愤怒、兴奋、恐惧、悲伤、惊讶、沮丧、高兴、失望和自然)。6373个视频用于训练，1775个视频用于验证，1807个视频用于测试。

### 复现精度

|                 | F1-Happy | F1-Sad | F1-Angry | F1-Neutral |
| --------------- | -------- | ------ | -------- | ---------- |
| LMF（论文精度） | 0.858    | 0.859  | 0.890    | 0.717      |
| LMF（复现精度） | 0.859    | 0.863  | 0.890    | 0.718      |


## 3. 快速使用

### 3.1 环境准备

实验环境
- Python：3.8.5
- PaddlePaddle：2.2.2

数据集下载

[下载链接](https://pan.baidu.com/s/1rg9Pgol9MG3EZyDlmHa6bA) ，提取码：ag52

将数据放在 data 文件夹中



### 3.2 结果验证


```bash
bash run_lfm.sh
```

输出示例：

```
Start testing happy......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
Model initialized
F1-score on test set is 0.8592420944673743
Accuracy score on test set is 0.8710021321961621

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
F1-score on test set is 0.8900815592777207
Accuracy score on test set is 0.8891257995735607

Start testing neutral......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
Model initialized
F1-score on test set is 0.7183055564744968
Accuracy score on test set is 0.7217484008528785
```

### 3.3 模型训练

```
train.sh
```

训练日志位于 train_log 文件夹下，模型存储在 check 文件夹下


### 3.4. 模型导出

export_model.py用于将训练模型动转静，静态模型保存到check_inference文件夹中。

happy、sad、angry、neutral的模型导出命令分别为：
```
python export_model.py --emotion happy --audio_hidden 4 --video_hidden 16 --text_hidden 128 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.5 --rank 1 --data_path ./data/iemocap.pkl
python export_model.py --emotion sad --audio_hidden 8 --video_hidden 4 --text_hidden 128 --audio_dropout 0 --video_dropout 0 --text_dropout 0 --rank 4 --data_path ./data/iemocap.pkl
python export_model.py --emotion angry --audio_hidden 8 --video_hidden 4 --text_hidden 64 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.15 --rank 8 --data_path ./data/iemocap.pkl
python export_model.py --emotion neutral --audio_hidden 32 --video_hidden 8 --text_hidden 64 --audio_dropout 0.2 --video_dropout 0.5 --text_dropout 0.2 --rank 16 --data_path ./data/iemocap.pkl
```

### 3.5. Inference推理
infer.py用于模型在未知数据上的推理测试，在当前分类任务中将预测概率结果保存到out_prob文件夹中。

happy、sad、angry、neutral的模型推理命令分别为：
```
python infer.py --emotion happy --data_path ./data/iemocap.pkl 
python infer.py --emotion sad --data_path ./data/iemocap.pkl
python infer.py --emotion angry --data_path ./data/iemocap.pkl
python infer.py --emotion neutral --data_path ./data/iemocap.pkl
```

### 3.6. TIPC自动化测试脚本

以Linux基础训练推理测试为例，我们准备了小型数据集sample_data.pkl（位于data文件夹中，由于容量限制，sample_data.pkl只包括happy和sad两种情绪，也可以在原始的iemocap数据集上运行）用来测试TIPC自动化脚本，具体见[README.md](/test_tipc/README.md)。

测试命令如下：

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/lmf/train_infer_python.txt lite_train_lite_infer
```

具体请参照[test_train_inference_python.md](./test_tipc/docs/test_train_inference_python.md)


## 4. 参考链接与文献

1. [Efficient Low-rank Multimodal Fusion with Modality-Specific Factors](https://arxiv.org/abs/1806.00064)
2. 参考repo：[Low-rank-Multimodal-Fusion](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)

