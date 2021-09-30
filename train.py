import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from model.LSTM import NER_LSTM
from model.LSTM_CRF import NER_LSTM_CRF
from model.BERT_LSTM_CRF import NER_BERT_LSTM_CRF
import config

class NERDataset(Dataset):
    def __init__(self, X, Y):
        # 真实数据
        self.data = [{'x': X[i], 'y': Y[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)   

def get_preds_and_labels(y_true, predict):
    """
     统计非0的，也就是真实标签的长度
    """
    leng = []
    preds, labels = [], []
    for item in y_true.cpu():
        tmp = []
        for j in item:
            if j.item()>0:
                tmp.append(j.item())
        leng.append(tmp)

    for index, item in enumerate(predict):
        preds += item[:len(leng[index])]

    for index, item in enumerate(y_true.tolist()):
        labels += item[:len(leng[index])] 

    return preds, labels


def train(model, model_name, train_dataloader, valid_dataloader):
    model = model.to(config.device)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)    

    for epoch in range(config.max_epoch):
        model.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_preds = []
        epoch_labels = []

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = batch['x']
            y = batch['y']

            X = X.to(config.device)
            y = y.to(config.device)

            predict = model(X)

            # 对数似然损失
            loss = model.log_likehood(X, y)

            epoch_loss += loss.item()
            loss.backward()

            preds, labels = get_preds_and_labels(y, predict)

            epoch_preds += preds
            epoch_labels += labels

            # 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            accuracy = accuracy_score(labels, preds)

            if step % 20 == 0:
                print('epoch:', epoch, 'step:', step, 'train loss:', round(loss.item(), 3), 
                    'train accuracy: ', round(accuracy, 3))

        epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
        epoch_loss = epoch_loss / len(train_dataloader)
        precision = precision_score(epoch_labels, epoch_preds, average='macro')
        recall = recall_score(epoch_labels, epoch_preds, average='macro')
        f1 = f1_score(epoch_labels, epoch_preds, average='macro')   

        print('epoch:', epoch, 'train loss:', round(epoch_loss, 3),  
            'train accuracy: ', epoch_accuracy)                     

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_preds = []
        epoch_labels = []

        best_accuracy = 0.0
        model.eval()

        for step, batch in enumerate(valid_dataloader):
            X = batch['x']
            y = batch['y']

            X = X.to(config.device)
            y = y.to(config.device)            

            predict = model(X)

            # 对数似然损失
            loss = model.log_likehood(X, y)

            epoch_loss += loss.item()   

            preds, labels = get_preds_and_labels(y, predict)

            epoch_preds += preds
            epoch_labels += labels        

            accuracy = accuracy_score(labels, preds)

            if step % 20 == 0:
                print('epoch:', epoch, 'step:', step, 'valid loss:', round(loss.item(), 3), 
                    'valid accuracy: ', round(accuracy, 3))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print('epoch:', epoch, 'step:', step, 'valid loss:', round(loss.item(), 3), 
                    'valid accuracy: ', round(accuracy, 3))
                print('Save best valid accuracy:', accuracy)
                torch.save(model.state_dict(), config.root_path + '/model/' 
                    + model_name.lower() + '.pkl') 

        epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
        epoch_loss = epoch_loss / len(valid_dataloader)
        precision = precision_score(epoch_labels, epoch_preds, average='macro')
        recall = recall_score(epoch_labels, epoch_preds, average='macro')
        f1 = f1_score(epoch_labels, epoch_preds, average='macro')   

        print('epoch:', epoch, 'valid loss:', round(epoch_loss, 3),
            'valid accuracy: ', round(epoch_accuracy, 3))   

if __name__ == '__main__':
    with open(config.pickle_path, 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train = pickle.load(inp)
            y_train = pickle.load(inp)
            x_test = pickle.load(inp)
            y_test = pickle.load(inp)
            x_valid = pickle.load(inp)
            y_valid = pickle.load(inp)

    print("train data len:",len(x_train))
    print("valid data len:",len(x_valid))

    train_dataset = NERDataset(x_train, y_train)
    valid_dataset = NERDataset(x_valid, y_valid)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=config.num_workers)

    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=config.num_workers) 

    models = {'NER_LSTM': NER_LSTM, 'NER_LSTM_CRF': NER_LSTM_CRF, 
            'NER_BERT_LSTM_CRF': NER_BERT_LSTM_CRF}

    model = models[config.model_name](embedding_dim=config.embedding_dim, 
                                        hidden_dim=config.hidden_dim,
                                        dropout=config.dropout,
                                        word2id=word2id,
                                        tag2id=tag2id)      
    train(model, config.model_name, train_dataloader, valid_dataloader)