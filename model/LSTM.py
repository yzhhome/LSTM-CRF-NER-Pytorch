import torch
from torch import nn
from torch.nn.modules.sparse import Embedding

class NER_LSTM(nn.Module):
    def __init__(self,
                embedding_dim,  # 词向量维度
                hidden_dim,     # 隐藏层大小
                dropout,        # dropout比率
                word2id,        # 词的id表示
                tag2id):        # 标签的id表示
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_idx = tag2id
        self.target_size = len(tag2id)
        
        self.word_embeds = nn.Embedding(self.embedding_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 因为是双向lstm，所以输出维度为self.hidden_dim // 2
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, 
            bidirectional=True, batch_first=True)

        # 接一个全连接层输出
        # 输入维度self.hidden_dim
        # 输出维度self.target_size
        self.hidden2tag = nn.Linear(self.hidden_dim, self.target_size)

    def forward(self, x):
        # word embedding
        embedding = self.word_embeds(x)

        # lstm输出
        outputs, hiddens = self.lstm(embedding)

        # dropput
        outputs = self.dropout(outputs)

        # 输出结果
        outputs = self.hidden2tag(outputs)
        return outputs