import pickle
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score
from model.LSTM import NER_LSTM
from model.LSTM_CRF import NERLSTM_CRF
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


def test(model, test_dataloader):
    model = model.to(config.device)
    print(model) 

    for epoch in range(config.max_epoch):              
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_preds = []
        epoch_labels = []

        model.eval()
        for step, batch in enumerate(test_dataloader):
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
                print('epoch:', epoch, 'step:', step, 'test loss:',  round(loss.item(), 3), 
                    'test accuracy: ', round(accuracy, 3))

        epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
        epoch_loss = epoch_loss / len(test_dataloader)
        precision = precision_score(epoch_labels, epoch_preds, average='macro')
        recall = recall_score(epoch_labels, epoch_preds, average='macro')
        f1 = f1_score(epoch_labels, epoch_preds, average='macro')   

        print('epoch:', epoch, 'test loss:', round(epoch_loss, 3),  
            'test accuracy: ', round(epoch_accuracy, 3))   

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

    print("test data len:",len(x_test))

    test_dataset = NERDataset(x_test, y_test)

    test_dataloader = DataLoader(dataset=test_dataset, 
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=config.num_workers)
    models = {'NERLSTM': NER_LSTM, 'NERLSTM_CRF': NERLSTM_CRF}

    model = models[config.model_name](embedding_dim=config.embedding_dim, 
                                        hidden_dim=config.hidden_dim,
                                        dropout=config.dropout,
                                        word2id=word2id,
                                        tag2id=tag2id)      
    model.load_state_dict(torch.load(config.root_path + '/model/' + 
        str(config.model_name).lower() + '.pkl'))
    test(model, test_dataloader)