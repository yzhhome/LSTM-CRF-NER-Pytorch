import torch
import os

root_path = os.path.abspath(os.path.dirname(__file__))
data_path = root_path + '/data/'
model_name = 'NER_BERT_LSTM_CRF'  

pickle_path = root_path + '/data/renmindata.pkl'    # 训练集存放路径
bert_path = '/home/opendata/temp/bert-base-chinese' # 加载bert预训练的模型的路径
bert_dim = 768

batch_size = 32  # batch size
num_workers = 4  # 加载数据使用的线程数
print_freq = 20  # 每隔多少个batch打印信息

max_epoch = 10
lr = 0.001      # 学习率
lr_decay = 0.5  # 学习率衰减参数
weight_decay = 1e-5  # 权重衰减

embedding_dim = 100  # 词向量维度
hidden_dim = 200     # 隐藏层大小
dropout = 0.2        # dropout比率

if torch.cuda.is_available():
    device = torch.device('cuda') 
else:
    device = torch.device('cpu') 