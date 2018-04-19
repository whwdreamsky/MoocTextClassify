# coding: utf-8
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
from gensim.models import KeyedVectors
#主要是序列化文本，并padding，使用预训练好的词向量
data = pd.read_csv('ques_ans.txt',sep='\t',header=None)
qus = list(data[0])
ans = list(data[0])
ans = list(data[1])
tk = Tokenizer()
tk.fit_on_texts(text_uniq)
vocabusize = len(tk.word_index) +1
ans_encode = tk.texts_to_sequences(ans)
qus_encode = tk.texts_to_sequences(qus)
y_lable = list(data[2])
dictionary = tk.word_index
dictionary_inv = dict((value,key) for key,value in dictionary.items())
dictionary_inv[0] ='PAD'
ans_encode_pad = sequence.pad_sequences(ans_encode,maxlen=100,padding="post",truncating="post")
ans_encode_pad[0]
qus_encode_pad = sequence.pad_sequences(qus_encode,maxlen=100,padding="post",truncating="post")
x_all = []
for i in range(len(qus)):
    tmp = []
    tmp.append(ans_encode_pad[i])
    tmp.append(qus_encode_pad[i])
    x_all.append(tmp)
shuffle_indices = np.random.permutation(np.arange(len(y_lable)))
x_all = x_all[shuffle_indices]
y_lable = np.array(y_lable)
y_lable = y_lable[shuffle_indices]
train_len = int(len(x_all)*0.9)
x_train = x_all[:train_len]
y_train = y_lable[:train_len]
x_test = x_all[train_len:]
y_test = y_lable[train_len:]
