import pickle
import torch
from model.LSTM import NER_LSTM
from model.LSTM_CRF import NERLSTM_CRF
from utils import get_tags, format_result
import config

def predict(model, tag, input_str=""):
    input_vec = [word2id.get(i, 0) for i in input_str]
    sentences = torch.tensor(input_vec).view(1, -1)
    paths = model(sentences)

    entities = []
    tags = get_tags(paths[0], tag, tag2id)
    entities += format_result(tags, input_str, tag)
    return entities

if __name__ == '__main__':
    with open(config.pickle_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)


    models = {'NERLSTM': NER_LSTM, 'NERLSTM_CRF': NERLSTM_CRF}

    model = models[config.model_name](embedding_dim=config.embedding_dim, 
                                        hidden_dim=config.hidden_dim,
                                        dropout=config.dropout,
                                        word2id=word2id,
                                        tag2id=tag2id)  

    model.load_state_dict(torch.load(config.root_path + '/model/' + 
        str(config.model_name).lower() + '.pkl'))

    for i in range(3):
        input_str = input("请输入文本: ")
        output = predict(model, 'nr', input_str)
        print(output)