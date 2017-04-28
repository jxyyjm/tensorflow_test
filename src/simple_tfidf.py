#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2016-12-18
# @author    : yujianmin
# @reference : 
# @what to-do: try a tf-idf model

'''
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer ## 将文本预料转换为n-gram表示，0,1向量.

corpus = [
	'This is the first document.',
	'this is the seconde document.',
	'And the third one.',
	'Is this the first document?',
]

## 初始化一个类 ##
vectorizer = CountVectorizer(min_df=1)
print 'type of vectorizer'
print type(vectorizer)

## 用预料库构造一个记录位置和值的类对象 ##
x = vectorizer.fit_transform(corpus)
print 'type of x '
print type(x)
print '1-gram-vectorizer'
print x.toarray()
## 列的含义 ##
vectorizer.get_feature_names()
## 特诊名称与列索引的转换保存在属性 vocabulary_里面 ##

## 对一个新预料也转成01向量 ## 但是之前预料库里没出现的会忽略掉 ##
analyze = vectorizer.build_analyzer()
analyze('this is a test document')

## 词的组合也想要表示出来 ## 1-grams and 2-grams
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
									token_pattern=r'\b\w+\b',
									min_df=1)
x = bigram_vectorizer.fit_transform(corpus)
print 'bigram-vectorizer'
print x.toarray()

analyze = bigram_vectorizer.build_analyzer()
print analyze('this is test-by me')

###############################################################
## TF-IDF 转换，将文本转换为 词频-逆向文档频率 ## 但不记录词的位置信息 ##
## 词的重要性 与 词在文件中出现的次数成 正比； 与 在语料库中出现的频率 成反比。##
## 一个词，如果在一篇文章中出现的频次很高，而在其他文章中出现的很少，认为其具有很好的类别区分能力 ##
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
print transformer

#   对某个词ti在文件dj中的TF-IDFij 计算 
#   TFij = Nij / sum(Nkj) 
#   TFij 表示某个词的词频 ，在某个文件中出现的频率 
#   Nij  表示某个词在文件dj中出现的次数
#   分母 表示某个词在文件dk中出现的次数总和 

#   IDFi = log{ |D| / (|包含词ti的文件数量|+1) }
#   |D|  表示语料的总文件数量 
#   分母 表示包含词ti的文件数量 + 1
#   TF-IDFij = TFij * IDFi

vectorizer  = CountVectorizer()
transformer = TfidfTransformer()
tfidf       = transformer.fit_transform(vectorizer.fit_transform(corpus))
keyword     = vectorizer.get_feature_names()
weight      = tfidf.toarray()
print 'keyword : '
print keyword
print 'tf-idf weight'
print weight
############################################################
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import jieba
import jieba.posseg as pseg
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

## 获取文件并分词处理 ##
def getFileAndSeg(dir): # -> []
	dir = dir.rstrip('/')
	corpus_text_list      = []
	corpus_file_name_list = os.listdir(dir)
	print 'current corpus names as following:'
	for file_name in corpus_file_name_list:
		file   = dir +'/'+ file_name
		print file
	for file_name in corpus_file_name_list:
		file   = dir +'/'+ file_name
		handle = open(file)
		text   = handle.read()
		seg    = jieba.cut(text, cut_all=True)
		seg_text = ' '.join(seg)
		handle.close()
		corpus_text_list.append(seg_text)
	return corpus_text_list, corpus_file_name_list
## 计算DF-IDF
def get_tfidf(corpus): # res_keyword=[]; res_matrix=np.array
	res_keyword= []
	res_matrix = np.array([])
	vectorizer = CountVectorizer()
	transformer= TfidfTransformer()
	tfidf      = transformer.fit_transform(vectorizer.fit_transform(corpus))
	res_matrix = tfidf.toarray()
	res_keyword= vectorizer.get_feature_names()
	return res_keyword, res_matrix
## 保存TF-IDF 结果  ##
def save_tfidf_weight(weight, corpus_names, keywords, filesave):
	if os.path.isfile(filesave): os.remove(filesave)
	handle = open(filesave, 'aw')
	col_names = 'corpus_name' +'\t' +'\t'.join(keywords)
	handle.write(col_names +'\n')
	print 'weight.shape     : ', weight.shape ## (3,67)
	print 'len(keywords)    : ', len(keywords)
	print 'len(corpus_name) : ', len(corpus_names)
	for index_corpus, corpus_name in enumerate(corpus_names):
		str_save = corpus_name
		for index_feature, feature_name in enumerate(keywords):
			#print 'index_feature : ', index_feature, 'index_corpus : ', index_corpus
			str_save += '\t'+str(weight[index_corpus, index_feature])
		handle.write(str_save +'\n')
	handle.close()

def main(dir, filesave):
	corpus, corpus_names  = getFileAndSeg(dir)
	keyword, tfidf_weight = get_tfidf(corpus)
	save_tfidf_weight(tfidf_weight,corpus_names, keyword, filesave)

if __name__=='__main__':
	corpus_dir = sys.argv[1]
	file_save  = sys.argv[2]
	main(corpus_dir, file_save)
	# run sample: python this.py ../data/ tmp.log #











