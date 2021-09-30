import codecs
import re
import pandas as pd
import numpy as np
import collections
import pickle
from sklearn.model_selection import train_test_split
from config import data_path

def originHandle():
    """
    将remin.txt删除前缀信息保存为remin2.txt    
    """
    with open(data_path +  'renmin.txt','r') as inp, \
        open(data_path + 'renmin2.txt','w') as outp:

        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i<len(line)-1:
                if line[i][0]=='[':
                    outp.write(line[i].split('/')[0][1:])
                    i+=1
                    while i<len(line)-1 and line[i].find(']')==-1:
                        if line[i]!='':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].strip()+'/'+line[i].split('/')[1][-2:]+' ')
                elif line[i].split('/')[1]=='nr':
                    word = line[i].split('/')[0] 
                    i+=1
                    if i<len(line)-1 and line[i].split('/')[1]=='nr':
                        outp.write(word+line[i].split('/')[0]+'/nr ')           
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i]+' ')
                i+=1
            outp.write('\n')

def originHandle2():
    """
    保留remin2.txt中的nr, ns, nt, 另存为remin3.txt    
    """    
    with codecs.open(data_path + 'renmin2.txt','r','utf-8') as inp, \
        codecs.open(data_path + 'renmin3.txt','w','utf-8') as outp:

        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                if tag=='nr' or tag=='ns' or tag=='nt':
                    outp.write(word[0]+"/B_"+tag+" ")
                    for j in word[1:len(word)-1]:
                        if j!=' ':
                            outp.write(j+"/M_"+tag+" ")
                    outp.write(word[-1]+"/E_"+tag+" ")
                else:
                    for wor in word:
                        outp.write(wor+'/O ')
                i+=1
            outp.write('\n')    

def sentence2split():
    """
    根据标点符号分隔句子
    """
    with open(data_path + 'renmin3.txt','r') as inp, \
        codecs.open(data_path + 'renmin4.txt','w','utf-8') as outp:

        texts = inp.read()
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)
        for sentence in sentences:
	        if sentence != " ":
		        outp.write(sentence.strip()+'\n')     

def data2pkl():
    """
    保存数据为pickle格式
    """
    datas = list()
    labels = list()
    linedata=list()
    linelabel=list()

    tags = set()
    tags.add('')

    input_data = codecs.open(data_path + 'renmin4.txt','r','utf-8')

    for line in input_data.readlines():
        line = line.split()
        linedata = []
        linelabel = []

        numNotO = 0
        for word in line:
            # 分隔每个词
            word = word.split('/')

            # 词
            linedata.append(word[0])

            # 词性
            linelabel.append(word[1])

            tags.add(word[1])

            if word[1]!='O':
                numNotO+=1
        
        # 不是O的词添加到data和label
        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()

    def flat_gen(x):
        def iselement(e):
            return not(isinstance(e, collections.Iterable) and not isinstance(e, str))
        for el in x:
            if iselement(el):
                yield el
            else:
                yield from flat_gen(el)   

    # 所有待识别的词
    all_words = [i for i in flat_gen(datas)]    
    
    sr_allwords = pd.Series(all_words)

    # 每个词的数量统计
    sr_allwords = sr_allwords.value_counts()    

    # 词的索引
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words)+1)    

    tags = [i for i in tags]
    tag_ids = range(len(tags))

    # 词转换为id表示
    word2id = pd.Series(set_ids, index=set_words)
    # id转换为词
    id2word = pd.Series(set_words, index=set_ids)

    # 标签转换为id表示
    tag2id = pd.Series(tag_ids, index=tags)
    # id转换为标签表示
    id2tag = pd.Series(tags, index=tag_ids)

    # 加上未知词
    word2id["unknow"]=len(word2id)+1
    id2word[len(word2id)]="unknow"

    max_len = 60
    def X_padding(words):
        """
        特征的padding操作
        """
        ids = list(word2id[words])
        if len(ids) >= max_len:  
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids

    def y_padding(tags):
        """
        标签的padding操作
        """
        ids = list(tag2id[tags])
        if len(ids) >= max_len: 
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))
    
    # 划分训练集和测试集
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)

    # 在训练集中再划分出验证集
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)

   # 用pickle保存数据
    with open(data_path + 'renmindata.pkl', 'wb') as outp:
	    pickle.dump(word2id, outp)
	    pickle.dump(id2word, outp)
	    pickle.dump(tag2id, outp)
	    pickle.dump(id2tag, outp)
	    pickle.dump(x_train, outp)
	    pickle.dump(y_train, outp)
	    pickle.dump(x_test, outp)
	    pickle.dump(y_test, outp)
	    pickle.dump(x_valid, outp)
	    pickle.dump(y_valid, outp)
    print('Finished saving the data.')

if __name__ == '__main__':    
    originHandle()
    originHandle2()
    sentence2split()
    data2pkl()
