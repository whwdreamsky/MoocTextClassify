# coding: utf-8
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.models import Sequential, Model,load_model
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras.utils.np_utils as np_utils
import numpy as np
from tools import load_data
from gensim.models import KeyedVectors


#主要是序列化文本，并padding，使用预训练好的词向量


sentence_maxlen = 100
num_filters =10
embedding_dim = 200
pooling_size =2
droprate = 0.5
batch_size = 12
num_epochs = 50
labledict=dict({"correct":0,"contradictory":1,"irrelevant":2,"non_domain":3,"partially_correct_incomplete":4})
sentence_maxlen1 = 30
sentence_maxlen2 = 70
def load_data_1(qus_ans_file,labeltype):
    # 这里label type twoway|fiveway
    data = pd.read_csv(qus_ans_file,sep='\t',header=None)
    qus = list(data[0])
    ans = list(data[1])
    y_lable = list(data[2])
    y_lable = [labledict[type] for type in y_lable]
    
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
    tk.fit_on_texts(text_uniq)
    vocabusize = len(tk.word_index) +1
    ans_encode = tk.texts_to_sequences(ans)
    qus_encode = tk.texts_to_sequences(qus)
    # 获取字典 dictionary_inv：(id : word) 这个是逆序的词典，用于之后过来查word2vec 词向量
    dictionary = tk.word_index
    dictionary_inv = dict((value,key) for key,value in dictionary.items())
    dictionary_inv[0] ='PAD'
    ans_encode_pad = sequence.pad_sequences(ans_encode,maxlen=sentence_maxlen,padding="post",truncating="post")
    qus_encode_pad = sequence.pad_sequences(qus_encode,maxlen=sentence_maxlen,padding="post",truncating="post")
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


def transToWord2Vec(x_train,word2vecweight,dictionary_inv):
    train_trans = []
    for sentence in x_train:
        sentence_trans = []
        for word in sentence:
            if dictionary_inv[word] in word2vecweight.vocab:
                sentence_trans.append(word2vecweight[dictionary_inv[word]])
            else:
                sentence_trans.append(np.zeros(word2vecweight.vector_size))
        train_trans.append(sentence_trans)
    return np.stack(train_trans)

def prepareInput(sentence):

    return
def predict(modelfile,test_file,dictionary_inv):
    return 





def trainModel(x_train,y_train,x_test,y_test,dictionary_inv,word2vecfile):
    word2vecweight = KeyedVectors.load_word2vec_format(word2vecfile,binary=True)
    #word2vecweight['PAD'] = np.zeros(word2vecweight.vector_size)
    qus_train = x_train[:,0]
    ans_train = x_train[:,1]
    qus_test = x_test[:,0]
    ans_test = x_test[:,1]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    # 这种是词向量固定 
    qus_train = transToWord2Vec(qus_train,word2vecweight,dictionary_inv)
    ans_train = transToWord2Vec(ans_train,word2vecweight,dictionary_inv)
    qus_test = transToWord2Vec(qus_test,word2vecweight,dictionary_inv)
    ans_test = transToWord2Vec(ans_test,word2vecweight,dictionary_inv)
    query = Input(name='query', shape=(sentence_maxlen,embedding_dim))
    doc = Input(name='doc', shape=(sentence_maxlen,embedding_dim))
    '''
    embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
    q_embed = embedding(query)
    show_layer_info('Embedding', q_embed)
    d_embed = embedding(doc)
    show_layer_info('Embedding', d_embed)
    '''
    q_conv1 = Convolution1D(filters=num_filters, kernel_size=3, padding='valid',activation="relu",strides=1) (query)
    d_conv1 = Convolution1D(filters=num_filters, kernel_size=3, padding='valid',activation="relu",strides=1) (doc)

    q_pool1 = MaxPooling1D(pool_size=pooling_size) (q_conv1)
    d_pool1 = MaxPooling1D(pool_size=pooling_size) (d_conv1)

    pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])

    pool1_flat = Flatten()(pool1)

    pool1_flat_drop = Dropout(rate=droprate)(pool1_flat)

    out_ = Dense(5, activation='softmax')(pool1_flat_drop)
    model = Model([query,doc], out_)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    model.fit([qus_train,ans_train], y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=([qus_test,ans_test], y_test), verbose=2)
    #存储模型
    model.save("model.h5")


x_train,y_train,x_test,y_test,dictionary_inv = load_data("../data/train_bettle/index_simple_token.txt","../data/train_bettle/relation.txt","../data/train_bettle/word.txt",labeltype="5way")
#x_train,y_train,x_test,y_test,dictionary_inv = load_data("./ques_ans.txt","")
trainModel(x_train,y_train,x_test,y_test,dictionary_inv,"./eng.vectors.bin")

