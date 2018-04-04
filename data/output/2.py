# coding: utf-8
from gensim import corpora
fsen = open("index_simple_token.txt",'r')
corpus = {}
for line in fsen:
    tmp = line.strip().split('\t')
    if len(tmp)<2:
        continue
    corpus[tmp[0]] =tmp[1].split(' ')
text = list(corpus.values())
dictionary= corpora.Dictionary(text)
#dictionary.save('dict.txt')
#fout = open('dict_txt.txt','w')
#for key,value in dictionary.token2id.items():
#    fout.write(key+'\t'+value+'\n')
    

#for key,value in dictionary.token2id.items():

#    fout.write(str(key)+'\t'+str(value)+'\n')
    
    
frealtion = open('relation.txt','r')
import pandas
question = []
answer = []
reference = []
accuracy = []
for line in frealtion:
    tmp = line.strip().split('\t')
    if len(tmp)<4:
        continue
    question.append(tmp[0])
    answer.append(tmp[1])
    reference.append(tmp[2])
    accuracy.append(tmp[3])
    
    
    
from gensim import corpora, models, similarities
