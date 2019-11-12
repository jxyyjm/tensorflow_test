#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')
import faiss
import numpy as np
import pandas as pd
import redis
r = redis.Redis(host='2', port=, db=, password='')

def load_key2emb(file_name):
  global d
  key_list = []
  emb_list = []
  handle = open(file_name, 'r')
  for line in handle:
    line = line.strip()
    if len(line) <= 0: continue
    words= line.split('\t')
    if len(words) != (d+1): continue
    key  = words[0]
    emb  = words[1:]
    emb  = [np.float32(i) for i in emb]
    key_list.append(key)
    emb_list.append(emb)
  print 'load from:',file_name, 'key.len=',len(key_list), \
          'emb.len=',len(emb_list), 'dim=', d
  return key_list, emb_list

d = 32
user_emb_file = sys.argv[1]
urls_emb_file = sys.argv[2]
user_list, user_emb = load_key2emb(user_emb_file)
urls_list, urls_emb = load_key2emb(urls_emb_file)

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, 10, faiss.METRIC_INNER_PRODUCT)
assert not index.is_trained
index.train(np.array(urls_emb))
index.add(np.array(urls_emb))
#index = faiss.IndexFlatL2(d)
#index.add(np.array(urls_emb))

for i in range(len(user_list)):
  user = user_list[i]
  emb  = user_emb[i]
  topK = 10
  D, I = index.search(np.array(emb, ndmin=2), topK)
  res_urls  = [urls_list[int(j)] for j in I[0]]
  res_score = D[0]
  for i in range(topK):
    md5 = res_urls[i]
    print user, md5, res_score[i], r.hget(md5, 'title')
  print '='*40

