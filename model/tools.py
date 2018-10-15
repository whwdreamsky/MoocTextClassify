# coding: utf-8
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.models import Sequential, Model,load_model
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.utils.np_utils as np_utils
import numpy as np
#from gensim.models import KeyedVectors
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0],self.validation_data[1]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return

labeldict=dict({"correct":0,"contradictory":1,"irrelevant":2,"non_domain":3,"partially_correct_incomplete":4})
labeldict_1=dict({"0":0,"1":1})

def load_word_dict(word_dict_file):
    word_data = pd.read_csv(word_dict_file,sep='\t',header=None,names=["word","index"])
    # dict 0 is PAD
    word_dict = {}
    word_dict_inverse = {}
    for index,row in word_data.iterrows():
        word_dict[row["word"]] = row["index"]
        word_dict_inverse[row["index"]] = row["word"]
    return word_dict,word_dict_inverse

def load_data(train_index_file,train_relation_file,test_index_file,test_relation_file,word_dict_file,labeltype,sentence_maxlen1=30,sentence_maxlen2=100):

    word_dict,word_dict_inverse = load_word_dict(word_dict_file)

    x_train,y_train = load_data_fromfile(train_index_file,train_relation_file,word_dict,labeltype,sentence_maxlen1,sentence_maxlen2)
    x_test,y_test = load_data_fromfile(test_index_file,test_relation_file,word_dict,labeltype,sentence_maxlen1,sentence_maxlen2)
    return x_train,y_train,x_test,y_test,word_dict_inverse
    

def load_data_fromfile(index_file,relation_file,word_dict,labeltype,sentence_maxlen1=30,sentence_maxlen2=100):
    # 这里label type twoway|fiveway
    index_data = pd.read_csv(index_file,sep='\t',header=None,names=["index","value"])
    #word_data = pd.read_csv(worddict,sep='\t',header=None,names=["word","index"])
    relation_data = pd.read_csv(relation_file,sep='\t',header=None,names=["question","response","answer","label"])
    query_dict = {}
    #word_dict = {}
    for index,row in index_data.iterrows():
        query_dict[row["index"]] = row["value"]
    
    #for index,row in word_data.iterrows():
    #    word_dict[row["word"]] = row["index"]
    y_lable = list(relation_data['label'])
    if labeltype =="twoway":
        y_twoway = []
        for item in y_lable:
            if item =="correct":
                y_twoway.append(labeldict["correct"])
            else:
                y_twoway.append(1)
        y_lable = y_twoway
    else:
        y_lable = [labledict[label] for label in y_lable]
    text_uniq = [str(item) for item in list(index_data["value"])]
    refer = []
    ans = []
    for index,row in relation_data.iterrows():
        ans.append(query_dict[row['response']])
        refer_str = ""
        for item in row['answer'].split(' '):
            #print(item)
            #for item in ansstr.split(' '):
            #    print(item)
            if item in query_dict:
                refer_str += query_dict[item]
        refer.append(refer_str)
    tk = Tokenizer()
    tk.word_index = word_dict
    #tk.fit_on_texts(text_uniq)
    vocabusize = len(tk.word_index) +1
    ans_encode = tk.texts_to_sequences(ans)
    refer_encode = tk.texts_to_sequences(refer)
    # 获取字典 dictionary_inv：(id : word) 这个是逆序的词典，用于之后过来查word2vec 词向量
    dictionary = tk.word_index
    dictionary_inv = dict((value,key) for key,value in dictionary.items())
    dictionary_inv[0] ='PAD'
    ans_encode_pad = sequence.pad_sequences(ans_encode,maxlen=sentence_maxlen1,padding="post",truncating="post")
    refer_encode_pad = sequence.pad_sequences(refer_encode,maxlen=sentence_maxlen2,padding="post",truncating="post")
    x_all = []
    for i in range(len(refer)):
        tmp = []
        tmp.append(ans_encode_pad[i])
        tmp.append(refer_encode_pad[i])
        x_all.append(tmp)
    # shuffle 数据
    x_all = np.array(x_all)
    y_lable = np.array(y_lable)
    #shuffle_indices = np.random.permutation(np.arange(len(y_lable)))
    #x_all = x_all[shuffle_indices]
    #y_lable = y_lable[shuffle_indices]
    #train_len = int(len(x_all)*splitrate)
    #x_train = x_all[:train_len]
    #y_train = y_lable[:train_len]
    #x_test = x_all[train_len:]
    #y_test = y_lable[train_len:]
    return x_all,y_lable


def load_data_1(qus_ans_file,word_dict_file,sentence_maxlen1 = 30,sentence_maxlen2 = 70,labeltype="fiveway"):
    # 这里label type twoway|fiveway

    word_dict,word_dict_inverse = load_word_dict(word_dict_file)
    data = pd.read_csv(qus_ans_file,sep='\t',header=None)
    qus = list(data[0])
    ans = list(data[1])
    y_lable = [str(item) for item in list(data[2])]
    y_lable = [labeldict_1[type1] for type1 in y_lable]
    
    if labeltype =="twoway":
        y_twoway = []
        for item in y_lable:
            if item =="correct":
                y_twoway.append(1)
            else:
                y_twoway.append(0)
        y_lable = y_twoway
    text = list(qus)
    text.extend(ans)
    text_uniq = list(set(text))
    tk = Tokenizer()
    tk.word_index = word_dict
    #tk.fit_on_texts(text_uniq)
    vocabusize = len(tk.word_index) +1
    ans_encode = tk.texts_to_sequences(ans)
    qus_encode = tk.texts_to_sequences(qus)
    # 获取字典 dictionary_inv：(id : word) 这个是逆序的词典，用于之后过来查word2vec 词向量
    dictionary = tk.word_index
    dictionary_inv = dict((value,key) for key,value in dictionary.items())
    #dictionary_inv[0] ='PAD'
    ans_encode_pad = sequence.pad_sequences(ans_encode,maxlen=sentence_maxlen1,padding="post",truncating="post")
    qus_encode_pad = sequence.pad_sequences(qus_encode,maxlen=sentence_maxlen2,padding="post",truncating="post")
    x_all = []
    for i in range(len(qus)):
        tmp = []
        tmp.append(ans_encode_pad[i])
        tmp.append(qus_encode_pad[i])
        x_all.append(tmp)
    # shuffle 数据
    x_all = np.array(x_all)
    shuffle_indices = np.random.permutation(np.arange(len(y_lable)))
    x_all = x_all[shuffle_indices]
    y_lable = np.array(y_lable)
    y_lable = y_lable[shuffle_indices]
    train_len = int(len(x_all)*0.8)
    x_train = x_all[:train_len]
    y_train = y_lable[:train_len]
    x_test = x_all[train_len:]
    y_test = y_lable[train_len:]
    return x_train,y_train,x_test,y_test,dictionary_inv



