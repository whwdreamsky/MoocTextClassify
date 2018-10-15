# coding: utf-8
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.models import Sequential, Model,load_model
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import backend as K
import keras.utils.np_utils as np_utils
import numpy as np
from tools import load_data,load_data_1
from tools import Metrics
from gensim.models import KeyedVectors


#主要是序列化文本，并padding，使用预训练好的词向量


sentence_maxlen = 100
num_filters =10
embedding_dim = 300
pooling_size =2
droprate = 0.5
batch_size = 12
num_epochs = 50
labledict=dict({"correct":0,"contradictory":1,"irrelevant":2,"non_domain":3,"partially_correct_incomplete":4})
sentence_maxlen1 = 30
sentence_maxlen2 = 70
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



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




def transdata(xlist):
    qus = np.zeros((xlist.shape[0],xlist[0].shape[0]))
    for i in range(xlist.shape[0]):
        qus[i] = xlist[i]
    return qus

def trainModel(x_train,y_train,x_test,y_test,dictionary_inv,word2vecfile):
    #word2vecweight = KeyedVectors.load_word2vec_format(word2vecfile,binary=False)
    # 作为PAD 的embeeding
    #UNK = np.random.randn(embedding_dim,)
    #PAD = np.random.randn(embedding_dim,)
    #embeedingweight = [PAD]
    #for i in range(1,len(dictionary_inv)):
    #    if dictionary_inv[i] in word2vecweight.wv.vocab:
    #        embeedingweight.append(word2vecweight[dictionary_inv[i]])
    #    else:
    #        embeedingweight.append(UNK)
    #embeedingweight = np.stack(embeedingweight)
    embeedingweight = np.load(word2vecfile)
    
    print(embeedingweight.shape)
    #word2vecweight['PAD'] = np.zeros(word2vecweight.vector_size)
    qus_train = transdata(x_train[:,0])
    ans_train = transdata(x_train[:,1])
    qus_test = transdata(x_test[:,0])
    ans_test = transdata(x_test[:,1])

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    # 这种是词向量固定 
    #qus_train = transToWord2Vec(qus_train,word2vecweight,dictionary_inv)
    #ans_train = transToWord2Vec(ans_train,word2vecweight,dictionary_inv)
    #qus_test = transToWord2Vec(qus_test,word2vecweight,dictionary_inv)
    #ans_test = transToWord2Vec(ans_test,word2vecweight,dictionary_inv)
    #query = Input(name='query', shape=(sentence_maxlen1,embedding_dim))
    #doc = Input(name='doc', shape=(sentence_maxlen2,embedding_dim))
    query = Input(name='query', shape=(sentence_maxlen1,))
    doc = Input(name='doc', shape=(sentence_maxlen2,))
    # 词向量不固定
    embedding = Embedding(len(dictionary_inv), embedding_dim, weights=[embeedingweight], trainable = True)
    q_embed = embedding(query)
    #show_layer_info('Embedding', q_embed)
    d_embed = embedding(doc)
    #show_layer_info('Embedding', d_embed)

    q_conv1 = Convolution1D(filters=num_filters, kernel_size=3, padding='valid',activation="relu",strides=1) (q_embed)
    d_conv1 = Convolution1D(filters=num_filters, kernel_size=3, padding='valid',activation="relu",strides=1) (d_embed)

    q_pool1 = MaxPooling1D(pool_size=pooling_size) (q_conv1)
    d_pool1 = MaxPooling1D(pool_size=pooling_size) (d_conv1)

    pool1 = Concatenate(axis=1) ([q_pool1, d_pool1])

    pool1_flat = Flatten()(pool1)

    pool1_flat_drop = Dropout(rate=droprate)(pool1_flat)

    out_ = Dense(2, activation='softmax')(pool1_flat_drop)
    metrics = Metrics()
    model = Model([query,doc], out_)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Train the model
    model.fit([qus_train,ans_train], y_train, batch_size=batch_size, epochs=num_epochs,validation_data=([qus_test,ans_test], y_test), verbose=2)
    #存储模型
    model.save("model.h5")


#x_train,y_train,x_test,y_test,dictionary_inv = load_data("../data/train_bettle/index_simple_token.txt","../data/train_bettle/relation.txt","../data/train_bettle/word.txt",labeltype="5way")
#x_train,y_train,x_test,y_test,dictionary_inv = load_data("./ques_ans.txt","")
#x_train,y_train,x_test,y_test,dictionary_inv = load_data("../data/train_bettle/index_simple_token.txt","../data/train_bettle/relation.txt","../data/test_bettle/ua_index_query.txt","../data/test_bettle/ua_relation.txt","../data/word.txt",labeltype="5way",sentence_maxlen1=30,sentence_maxlen2=70)
#trainModel(x_train,y_train,x_test,y_test,dictionary_inv,"/Users/oliver/workplace/deeplearning/resource/wiki.en.vec")

x_train,y_train,x_test,y_test,dictionary_inv = load_data_1("../data/wiki/WikiQA-train.txt","../data/wiki/word.txt",labeltype="5way")
trainModel(x_train,y_train,x_test,y_test,dictionary_inv,"/Users/oliver/workplace/deeplearning/resource/wiki_sgns_2.npy")
