# coding: utf-8
get_ipython().run_line_magic('run', '2.py')
text[0]
corpus_index = {}
for key,value in corpus.items():
    corpus_index [key] = dictionary.doc2bow(value)
    
corpus_index[0]
corpus_index["answer204"]
lsi = models.LsiModel(corpus_index,id2word=dictionary,num_topics=6)
tfidf = models.TfidfModel(corpus_index)
get_ipython().run_line_magic('ls', '')
tfidf = models.TfidfModel(corpus_index,dictionary=dictionary)
tfidf = models.TfidfModel(corpus_index)
tfidf[corpus_index['answer204']]
q1 = corpus_index[question[0]]
a1 = corpus_index[answer[0]]
r1 = corpus_index[reference[0]]
accuracy[0]
q1
tfidf = models.TfidfModel(corpus_index)
corpus_index[0]
tfidf = models.TfidfModel(list(corpus_index.values()))
tfidf[q1]
tfidf[a1]
tfidf[r1]
answer[0]
tfidf[q1]
tfidf_corpus = tfidf[corpus_index]
tfidf_corpus = tfidf[list(corpus_index.values())]
lsi = models.LsiModel(list(corpus_index.values()),id2word=dictionary,num_topics=100)
lsi.save('LSI.pkl')
index = similarities.MatrixSimilarity(lsi[corpus])
index = similarities.MatrixSimilarity(lsi[list(corpus_index.values())])
a1
a1_lsi = lsi[a1]
simi = index[a1_lsi]
sort_sims = sorted(enumerate(simi),key= lambda item: -item[1])
print sort_sims[0:10]
print (sort_sims[0:10])
get_ipython().run_line_magic('save', '1-38 gensim.py')
