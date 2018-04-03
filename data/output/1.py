# coding: utf-8
import nltk
fsen = open('index_query.txt','r')
corpus = {}
for line in fsen:
    tmp = line.strip().split('\t')
    if len(tmp)<2:
        continue
    words = [word.lower() for word in nltk.word_tokenize(tmp[1]) ]
    corpus[tmp[0]] = words
    
