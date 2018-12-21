#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys 
import six 

class Test(object):
  def __init__(self, a=2, b=2):
    self.a = a 
    self.b = b 
  @classmethod
  def from_dict(cls, json_ab):
    test = Test()
    for (key, value) in six.iteritems(json_ab):
      test.__dict__[key] = value
    return test

  @classmethod
  def from_dict2(cls, json_ab):
    return cls.from_dict(json_ab)
  ## notice: classmethod 的精髓在于是不用实例化类，即可调用 类内函数 ## cls ==self ##
  ## from_dict 会返回初始化之后的类实例 ##

  def print_para(self):
    print 'self.a:\t', self.a
    print 'self.b:\t', self.b

if __name__=="__main__":
  z = Test.from_dict({'a':100, 'b':200})
  z.print_para()

