### LSTM+CRF实现序列标注

#### 代码结构介绍

`data`: 数据存放目录

`model/LSTM.py`: 双向LSTM模型实现命名实体被别

`model/LSTM_CRF.py`: 双向LSTM+CRF模型实现命名实体被别

`model/BERT_LSTM_CRF.py`: BERT+双向LSTM+CRF模型实现命名实体被别

`config.py`: 项目的一些配置参数

`dataprocess.py`: 处理原始数据

`utils.py`: 项目用的一些函数

`train.py`: 模型训练

`test.py`: 模型测试

`predict.py`: 模型预测