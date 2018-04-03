# coding: utf-8
get_ipython().run_line_magic('run', '1.py')
a = corpus.values()
a = list(a)
a[0]
fout = open('index_simple_token.txt','w')
for key,value in corpus.items():
    fout.write(key+'\t'+' '.join(value)+'\n')
    
fout.close()
from gensim import corpora
dictionary = corpora.Dictionary(a)
dictionaryl.save('dict.txt')
dictionary.save('dict.txt')
fout = open('dict_txt.txt','w')
for key,value in dictionary.token2id.items():
    fout.write(key+'\t'+value+'\n')
    
for key,value in dictionary.token2id.items():
    fout.write(str(key)+'\t'+str(value)+'\n')
    
    
dictionary.doc2bow(a[0])
type(a[0])
type(a[0][0])
a[0][1]
a[0][0]
b =dictionary.doc2bow(a[0])
type(b[0])
b[0]
b[0][0]
b[0][1]
frealtion = open('relation.txt','r')
import pandas
question = []
answer = []
reference = []
accuracy = []
i=0
for line in frealtion:
    tmp = line.strip().split('\t')
    if len(tmp<4):
        continue
    question[i] = tmp[0]
    answer[i] = tmp[1]
    reference[i] = tmp[2]
    accuracy[i] = tmp[3]
    
    
question = []
answer = []
reference = []
accuracy = []
i=0
for line in frealtion:
    tmp = line.strip().split('\t')
    if len(tmp)<4:
        continue
    question[i] = tmp[0]
    answer[i] = tmp[1]
    reference[i] = tmp[2]
    accuracy[i] = tmp[3]
    
    
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
    
    
    
len(question)
question[0]
reference[0]
accuracy[0]
answer[0]
corpus[question[0]]
corpus[answer[0]]
import pandas
from gensim import corpora, models, similarities
get_ipython().run_line_magic('save', '2.py 1-39')
