# Linux GPU/CPU 基础训练推理测试

Linux GPU/CPU 基础训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 |
|  :----: |   :----:  |    :----:  |
|  Efficient Low-rank Multimodal Fusion with Modality-Specific Factors  | LMF | 正常训练 | 


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  Efficient Low-rank Multimodal Fusion with Modality-Specific Factors   |  LMF |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据

用于基础训练推理测试的数据位于`./data/sample_data.pkl`。


### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.0
    ```

- 安装AutoLog（规范化日志输出工具）
    ```
    pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

### 2.3 功能测试


测试命令如下：

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/lmf/train_infer_python.txt lite_train_lite_infer
```

输出结果如下：

```bash
Epoch 0 complete! Average Training loss: 0.6811608409881592
Validation loss is: 0.6605233192443848
Found new check model, saving to disk...
Epoch 1 complete! Average Training loss: 0.6064284121990203
Validation loss is: 0.5399923324584961
Found new check model, saving to disk...
Epoch 2 complete! Average Training loss: 0.4072037336230278
Validation loss is: 0.3429754972457886
Found new check model, saving to disk...
Epoch 3 complete! Average Training loss: 0.3288413551449776
Validation loss is: 0.3257713317871094
Found new check model, saving to disk...
Epoch 4 complete! Average Training loss: 0.3574151992797851
Validation loss is: 0.360327935218811
F1-score on test set is 0.7810810810810811
Accuracy score on test set is 0.85
 Run successfully with command - python3.8 train.py --emotion happy --model_path=./check --epochs=5   --batch_size=16 --data_path ./data/sample_data.pkl!  

Start testing happy......
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
W0421 00:20:40.299171 12155 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 11.0
W0421 00:20:40.300539 12155 device_context.cc:465] device: 0, cuDNN Version: 8.0.
Model initialized
F1-score on test set is 0.7810810810810811
Accuracy score on test set is 0.85
 Run successfully with command - python3.8 test.py --data_path ./data/sample_data.pkl  !  

Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
W0421 00:20:45.376390 12198 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 11.0
W0421 00:20:45.377681 12198 device_context.cc:465] device: 0, cuDNN Version: 8.0.
inference model has been saved into ./check_inference/
 Run successfully with command - python3.8 ./export_model.py    !  
 
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
F1-score on test set is 0.7810810810810811
Accuracy score on test set is 0.85
[2022/04/21 00:20:53] root INFO: 

[2022/04/21 00:20:53] root INFO: ---------------------- Env info ----------------------
[2022/04/21 00:20:53] root INFO:  OS_version: Ubuntu 18.04
[2022/04/21 00:20:53] root INFO:  CUDA_version: 11.0.194
Build cuda_11.0_bu.TC445_37.28540450_0
[2022/04/21 00:20:53] root INFO:  CUDNN_version: None.None.None
[2022/04/21 00:20:53] root INFO:  drivier_version: 450.51.05
[2022/04/21 00:20:53] root INFO: ---------------------- Paddle info ----------------------
[2022/04/21 00:20:53] root INFO:  paddle_version: 2.2.2
[2022/04/21 00:20:53] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/04/21 00:20:53] root INFO:  log_api_version: 1.0
[2022/04/21 00:20:53] root INFO: ----------------------- Conf info -----------------------
[2022/04/21 00:20:53] root INFO:  runtime_device: gpu
[2022/04/21 00:20:53] root INFO:  ir_optim: True
[2022/04/21 00:20:53] root INFO:  enable_memory_optim: True
[2022/04/21 00:20:53] root INFO:  enable_tensorrt: False
[2022/04/21 00:20:53] root INFO:  enable_mkldnn: False
[2022/04/21 00:20:53] root INFO:  cpu_math_library_num_threads: 1
[2022/04/21 00:20:53] root INFO: ----------------------- Model info ----------------------
[2022/04/21 00:20:53] root INFO:  model_name: LMF
[2022/04/21 00:20:53] root INFO:  precision: fp32
[2022/04/21 00:20:53] root INFO: ----------------------- Data info -----------------------
[2022/04/21 00:20:53] root INFO:  batch_size: 1
[2022/04/21 00:20:53] root INFO:  input_shape: dynamic
[2022/04/21 00:20:53] root INFO:  data_num: 1
[2022/04/21 00:20:53] root INFO: ----------------------- Perf info -----------------------
[2022/04/21 00:20:53] root INFO:  cpu_rss(MB): 2056.5156, gpu_rss(MB): 6641.0, gpu_util: 98.0%
[2022/04/21 00:20:53] root INFO:  total time spent(s): 2.1586
[2022/04/21 00:20:53] root INFO:  preprocess_time(ms): 10.8461, inference_time(ms): 2146.8964, postprocess_time(ms): 0.8891
 Run successfully with command - python3.8 ./infer.py --data_path ./data/sample_data.pkl --save_inference_dir ./check_inference --use-gpu=True   --batch_size=1   --benchmark=True > ./test_tipc/logs/python_infer_gpu_batchsize_1.log 2>&1 !  

python3.8 ./infer.py --data_path ./data/sample_data.pkl --save_inference_dir ./check_inference --use-gpu=False --batch_size=1 --benchmark=True > ./test_tipc/logs/python_infer_cpu_batchsize_1.log 2>&1 
Audio feature dimension is: 74
Visual feature dimension is: 35
Text feature dimension is: 300
F1-score on test set is 0.7810810810810811
Accuracy score on test set is 0.85
[2022/04/21 00:20:58] root INFO: 

[2022/04/21 00:20:58] root INFO: ---------------------- Env info ----------------------
[2022/04/21 00:20:58] root INFO:  OS_version: Ubuntu 18.04
[2022/04/21 00:20:58] root INFO:  CUDA_version: 11.0.194
Build cuda_11.0_bu.TC445_37.28540450_0
[2022/04/21 00:20:58] root INFO:  CUDNN_version: None.None.None
[2022/04/21 00:20:58] root INFO:  drivier_version: 450.51.05
[2022/04/21 00:20:58] root INFO: ---------------------- Paddle info ----------------------
[2022/04/21 00:20:58] root INFO:  paddle_version: 2.2.2
[2022/04/21 00:20:58] root INFO:  paddle_commit: b031c389938bfa15e15bb20494c76f86289d77b0
[2022/04/21 00:20:58] root INFO:  log_api_version: 1.0
[2022/04/21 00:20:58] root INFO: ----------------------- Conf info -----------------------
[2022/04/21 00:20:58] root INFO:  runtime_device: cpu
[2022/04/21 00:20:58] root INFO:  ir_optim: True
[2022/04/21 00:20:58] root INFO:  enable_memory_optim: True
[2022/04/21 00:20:58] root INFO:  enable_tensorrt: False
[2022/04/21 00:20:58] root INFO:  enable_mkldnn: False
[2022/04/21 00:20:58] root INFO:  cpu_math_library_num_threads: 1
[2022/04/21 00:20:58] root INFO: ----------------------- Model info ----------------------
[2022/04/21 00:20:58] root INFO:  model_name: LMF
[2022/04/21 00:20:58] root INFO:  precision: fp32
[2022/04/21 00:20:58] root INFO: ----------------------- Data info -----------------------
[2022/04/21 00:20:58] root INFO:  batch_size: 1
[2022/04/21 00:20:58] root INFO:  input_shape: dynamic
[2022/04/21 00:20:58] root INFO:  data_num: 1
[2022/04/21 00:20:58] root INFO: ----------------------- Perf info -----------------------
[2022/04/21 00:20:58] root INFO:  cpu_rss(MB): 998.1992, gpu_rss(MB): None, gpu_util: None%
[2022/04/21 00:20:58] root INFO:  total time spent(s): 3.3108
[2022/04/21 00:20:58] root INFO:  preprocess_time(ms): 8.4269, inference_time(ms): 3301.46, postprocess_time(ms): 0.9339
 Run successfully with command - python3.8 ./infer.py --data_path ./data/sample_data.pkl --save_inference_dir ./check_inference --use-gpu=False --batch_size=1 --benchmark=True > ./test_tipc/logs/python_infer_cpu_batchsize_1.log 2>&1 !  ```