#!/usr/bin/python                                                                                                                                                                              
# -*- coding:utf-8 -*-

import os
import sys
import math
import numpy as np
import pandas as pd

smat = [[1.0, 0.8, 0.6, 0.9, 0.4], \
        [0.8, 1.0, 0.5, 0.7, 0.3], \
        [0.6, 0.5, 1.0, 0.4, 0.2], \
        [0.9, 0.7, 0.4, 1.0, 0.1], \
        [0.4, 0.3, 0.2, 0.1, 1.0]]
rank = [0.9, 0.8, 0.7, 0.6, 0.5]
rmat = [[0.81, 0.72, 0.63, 0.54, 0.45], \
        [0.72, 0.64, 0.56, 0.48, 0.40], \
        [0.63, 0.56, 0.49, 0.42, 0.35], \
        [0.54, 0.48, 0.42, 0.36, 0.30], \
        [0.45, 0.40, 0.35, 0.30, 0.25]]
ran_matrix = np.array(rmat, dtype=np.float32)
sim_matrix = np.array(smat, dtype=np.float32)
cand_len   = len(ran_matrix)
#print 'rank', rank
#print 'sim', smat
def get_kernel_matrix(ran_matrix, sim_matrix):
  ker_matrix = np.array([[0.0 for i in range(cand_len)] for i in range(cand_len)])
  for i in range(cand_len):
    for j in range(cand_len):
      ker_matrix[i][j] = ran_matrix[i][j] * sim_matrix[i][j]
  return ker_matrix

kernel_matrix = get_kernel_matrix(ran_matrix, sim_matrix)
print 'kernel_matrix\n', kernel_matrix
'''
1: Input: Kernel L, stopping criteria
2: Initialize: ci = [], di^2= L ii , j = argmax i∈Z log(di^2 ), Y g = {j}
3: while stopping criteria not satisfied do
4:   for i ∈ Z \ Y g do
5:     e i = (L ji − (c j ,c i))/d j
6:     c i = [c i, e i ], di^2= di^2− ei^2
7:   end for
8:   j = argmax i∈Z\Y g log(di^2 ), Y g = Y g ∪ {j}
9: end while
10: Return: Y g
'''
def func(kernel_matrix, N):
  c_matrix = np.array([[0.0 for i in range(len(kernel_matrix))] for i in range(len(kernel_matrix))])
  candidate_list = [0, 1, 2, 3, 4]
  j_opt   = 0
  result  = [j_opt]
  while len(result) < N:
    dj_opt  = math.sqrt(kernel_matrix[j_opt][j_opt])
    print 'iter=', len(result), '='*20, 'res=', result, 'j_opt=', j_opt, 'dj_opt^2=', round(dj_opt*dj_opt, 4)
    for i in candidate_list:
      if i in result: continue
      ei = (kernel_matrix[j_opt][i] - np.dot(c_matrix[j_opt], c_matrix[i])) / dj_opt
      ## here np.dot is to compute the opt-j with candidate-i ## could also done by as follow line ##
      ## ei = (kernel_matrix[j_opt][i] - np.dot(c_matrix[j_opt][0:len(result)], c_matrix[i][0:len(result)])) / dj_opt
      c_matrix[i][len(result)-1] = ei ## append the new ei ; is also update the zero to a value ##
      kernel_matrix[i][i] -= ei*ei
    if len(result) >= len(kernel_matrix): break
    #print 'c_matrix\n', c_matrix
    #print 'k_matrix\n', kernel_matrix
    tmp = 0
    for i in candidate_list:
      if i in result: continue
      if kernel_matrix[i][i] > tmp:
        j_opt = i
        tmp   = kernel_matrix[i][i]
    result.append(j_opt)
  return result

res = func(kernel_matrix, 8)
print 'res', res
