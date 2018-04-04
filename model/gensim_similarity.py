#coding: utf-8

from gensim import corpora,models,similarities
import pandas as pd
import sys

def loadCorpus(file_token,file_dict,relation_file):
	fsen = open(file_token,'r')
	# 读取分词完文本
	corpus_token = {}
	for line in fsen:
		tmp = line.strip().split('\t')
		if len(tmp)<2:
			continue
		corpus_token[tmp[0]] = tmp[1].split(' ')
	#导入词典
	dictionary = corpora.Dictionary.load(file_dict)
	# 序列化文本
	corpus_index = {}
	for key,value in corpus_token.items():
		corpus_index[key] = dictionary.doc2bow(value)
	#读取关系表
	relationdata = pd.read_csv(relation_file,sep='\t')

	return corpus_token,corpus_index,relationdata,dictionary

def outputresult(question,realtions,qu_simi,refersimi,fout):
	
	for i in range(len(realtions)):
		str1 = "\t".join(realtions.iloc[i])
		fout.write(str1+'\t'+str(qu_simi[i][1])+'\t'+str(refersimi[i][1])+'\n')
	

def LsiSimilari(corpus_token,corpus_index,relationdata,outputfile,dictionary):
	fout = open(outputfile,'w')
	text = list(corpus_index.values())
	tfidf = models.TfidfModel(text)
	questions = list(set(relationdata['Question']))
	for question in questions:
		realtions = relationdata[relationdata['Question']==question]

		reference = list(set(realtions['Answer']))
		answer = list(realtions['Response'])
		# 对一个question的所有句子建立LSI 模型计算 相似度，这里每个相似度计算式 quesiton 对 
		
		answer_seq = [ corpus_index[seq_id] for seq_id in answer]

		lsi = models.LsiModel(answer_seq,id2word=dictionary,num_topics=100)
		index = similarities.MatrixSimilarity(lsi[answer_seq])

		# 计算quesion的相对这个回复的相似度
		quesiton_seq = corpus_index[question]
		vec_lsi = lsi[quesiton_seq]
		qu_simi = list(enumerate(index[vec_lsi]))
		# 计算标准答案相对回复的相似度 这里针对多个标准答案我们取最大值
		refersimi = []
		refersimi_max = []
		reference_seq = [ corpus_index[seq_id] for seq_id in reference ]
		for seq in reference_seq:
			vec_lsi = lsi[seq]
			simi = list(enumerate(index[vec_lsi]))
			refersimi.append(simi)
		for i in range(len(simi)):
			maxnum = -100
			for j in range(len(refersimi)):
				# 因为这里每个元素是一个tuple
				if refersimi[j][i][1]>maxnum:
					maxnum = refersimi[j][i][1]
			tmp = (i,maxnum)
			refersimi_max.append(tmp)

		# 输出
		outputresult(question,realtions,qu_simi,refersimi_max,fout)
	fout.close()


if __name__ == '__main__':
	file_token,file_dict,relation_file,outputfile= sys.argv[1:]
	corpus_token,corpus_index,relationdata,dictionary = loadCorpus(file_token,file_dict,relation_file)
	LsiSimilari(corpus_token,corpus_index,relationdata,outputfile,dictionary)
