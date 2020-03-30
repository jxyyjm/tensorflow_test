#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')
import faiss
import numpy as np
import pandas as pd
from DataRead import getNow

'''
d = 64      ## dimension ##
nlist = 100 ## cluster nums ##
k = 4 ## find the top-k sim ##
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
# here we specify METRIC_L2, by default it performs inner-product search

assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
'''
def load_id2emb(file_name):
  res = []
  handle = open(file_name, 'r')
  for line in handle:
    line = line.strip()
    if len(line) <= 0: continue
    words= line.split('\t')
    if len(words) != 8: continue
    emb  = words
    emb  = [np.float32(i) for i in emb]
    res.append(emb)
  handle.close()
  print 'load user emb', len(res), 'from:', file_name
  return np.asarray(res)

def load_user_emb(file_name):
  res = {}
  handle = open(file_name, 'r')
  for line in handle:
    line = line.strip()
    if len(line) <= 0: continue
    words= line.split('\t')
    if len(words) != 9: continue
    user = words[0]
    emb  = words[1:]
    emb  = np.array([np.float32(i) for i in emb]).reshape((1,8))
    res[user] = emb
  handle.close()
  print 'load user emb', len(res), 'from:', file_name
  return res
def load_kv(file_name):
  res = {}
  res_reverse = {}
  handle = open(file_name, 'r')
  for line in handle:
    line = line.strip()
    if len(line) <= 0: continue
    words= line.split('\t')
    if len(words) != 2: continue
    key  = words[0]
    value= words[1]
    res[key] = value
    res_reverse[value] = key
  handle.close()
  print 'load kv ', len(res), 'from:', file_name
  return res, res_reverse

d = 8
nlist = 100
#id2emb = pd.read_table('./id2emb', sep='\t', names=['ktag_'+str(i) for i in range(8)], dtype=np.float32).values
#id2emb = load_id2emb('../data/ktag.emb')
#tag2id, id2tag = load_kv('../data/ktag.newid')
#user_emb_map = load_user_emb('../data/user2emb')
id2emb = load_id2emb(sys.argv[1])
tag2id, id2tag = load_kv(sys.argv[2])
user_emb_map = load_user_emb(sys.argv[3])
file_save = sys.argv[4]
try: topK = int(float(sys.argv[5]))
except: topK = 30

index  = faiss.IndexFlatIP(d)
index.add(id2emb)

#quantizer    = faiss.IndexFlatL2(d)
#index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
#assert not index.is_trained
#index.train(np.ascontiguousarray(id2emb))
#index.train(id2emb)
#index.add(id2emb)
#assert index.is_trained

import time
if os.path.isfile(file_save)==True:
  os.rename(file_save, file_save +'.'+getNow())
handle = open(file_save, 'aw')

for user, user_emb in user_emb_map.items():
  #print 'user', user, 'user_emb', user_emb
  S, I = index.search(user_emb, topK)
  #print 'type(I)', I, I.shape
  Id   = [str(i) for i in I[0]]
  Tag  = [id2tag[i] for i in Id]
  Score= [str(i) for i in S[0]]
  res  = [Tag[i]+'\2'+Score[i] for i in range(topK)]
  handle.write(user +'\t'+ '\1'.join(res) +'\n')

handle.close()
