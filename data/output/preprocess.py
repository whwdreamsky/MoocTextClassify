# coding: utf-8
from gensim import corpora
fsentence = open('index_query.txt','r')
corpus = {}
for line in fsentence:
    tmp = line.strip().split('\t')
    corpus[tmp[0]] = tmp[1]
    
len(corpus)
a = corpus.values
len(a)
a = corpus.values()
len(a)
type(a)
a = list(corpus.values())
dictionary = corpora.Dictionary(a)
import nltk
texts = nltk.word_tokenize(a)
corpus_index = {}
a[0]
nltk.word_tokenize(a[0])
for key,value in corpus.items():
    a = nltk.word_tokenize(str(value))
    
b = nltk.word_tokenize(a[0])
c = [x.lower() for x in b if not in nltk.corpus.stopwords]
c = [word.lower() for word in b if word not in nltk.corpus.stopwords]
stoplist = nltk.corpus.stopwords
type(stoplist)
stoplist = list(nltk.corpus.stopwords)
stoplist
stoplist.words('english')
stoplist.words('chinese')
c = [word.lower() for word in b if word not in nltk.corpus.stopwords.words('english')]
c = [word.lower() for word in a if word not in nltk.corpus.stopwords.words('english')]
c
for key,value in corpus.items():
    words = nltk.word_tokenize(str(value))
    c = [word.lower() for word in words if word not in nltk.corpus.stopwords.words('english')]
    corpus_index[key] = c
    
    
fout = open('index_tokenized.txt','w')
for key,value in corpus_index.items():
    fout.write(key+'\t'+value+'\n')
    
for key,value in corpus_index.items()
    fout.write(key+'\t'+" ".join(value)+'\n')

    
for key,value in corpus_index.items():
    fout.write(key+'\t'+" ".join(value)+'\n')
    

    
fout.close()
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
for key,value in corpus_index.items():
    words = [word for word in value if not in english_punctuations]
    fout.write(key+'\t'+" ".join(words)+'\n')

    

    
fout = open('index_tokenized.txt','w')
for key,value in corpus_index.items():
    words = [word for word in value if word not in english_punctuations]
    fout.write(key+'\t'+" ".join(words)+'\n')
    

    

    
fout.close()
fin = open('index_tokenized.txt','r')
for line in fin:
    tmp  = line.strip().split('\t')
    corpus_index[tmp[0]] = tmp[1]
    
for line in fin:
    tmp  = line.strip().split('\t')
    if len(tmp)<2:
        continue
    corpus_index[tmp[0]] = tmp[1]
    
    
len(corpus)
len(corpus_index)
dictionary = corpora.Dictionary(list(corpus_index.values))
dictionary = corpora.Dictionary(list(corpus_index.values()))
for line in fin:
    tmp  = line.strip().split('\t')
    if len(tmp)<2:
        continue
    words = tmp[1].split(' ')
    corpus_index[tmp[0]] = words
    
    
    
dictionary = corpora.Dictionary(list(corpus_index.values()))
a = corpus_index.values()
a[0]
a = list(corpus_index.values())
a[0]
get_ipython().run_line_magic('save', 'preprocess.py 1-60')
