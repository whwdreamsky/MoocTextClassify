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




labledict=dict({"correct":0,"contradictory":1,"irrelevant":2,"non_domain":3,"partially_correct_incomplete":4})

def load_data(train_index_file,train_relation_file,test_index_file,test_relation_file,word_dict_file,labeltype,sentence_maxlen1=30,sentence_maxlen2=100):
    word_data = pd.read_csv(word_dict_file,sep='\t',header=None,names=["word","index"])
    # dict 0 is PAD
    word_dict = {}
    word_dict_inverse = {}
    for index,row in word_data.iterrows():
        word_dict[row["word"]] = row["index"]
        word_dict_inverse[row["index"]] = row["word"]
        
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
