import torch
from torch import nn
from torchcrf import CRF
from transformers import BertModel
from config import bert_path

class NER_BERT_LSTM_CRF(nn.Module):
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

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # 因为是双向lstm，所以输出维度为self.hidden_dim // 2
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=False)

        # 接一个全连接层输出
        # 输入维度self.hidden_dim
        # 输出维度self.target_size
        self.hidden2tag = nn.Linear(self.hidden_dim, self.target_size)
        self.crf = CRF(self.target_size, batch_first=False)

        # 预训练模型加载Bert
        self.bert = BertModel.from_pretrained(bert_path)

        # Bert输出是768维，必须降维
        self.bert_linear = nn.Linear(768, self.embedding_dim)
    
    def _bert_enc(self, x):
        with torch.no_grad():
            encoded_layer = self.bert(x)
            encoded = encoded_layer[0]
            
            # 将768维降为100维
            encoded = self.bert_linear(encoded)
            return encoded

    def forward(self, x):
        x = x.transpose(0,1)

        # word embedding
        embedding = self._bert_enc(x)

        # lstm输出
        outputs, hiddens = self.lstm(embedding)

        # dropout
        outputs = self.dropout(outputs)

        # 降维输出
        outputs = self.hidden2tag(outputs)

        # crf中解码
        outputs = self.crf.decode(outputs)
        return outputs

    def log_likehood(self, x, tags):
        """
        对数似然函数值
        """
        x = x.transpose(0, 1)
        tags = tags.transpose(0, 1)

        embedding = self._bert_enc(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)

        return - self.crf(outputs, tags)