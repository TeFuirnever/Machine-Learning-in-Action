#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:58
# @Author  : GXl
# @File    : 3.5.2.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import pickle

"""
Parameters:
    filename - 决策树的存储文件名
Returns:
    pickle.load(fr) - 决策树字典
"""
# 函数说明:读取决策树
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    myTree = grabTree('classifierStorage.txt')
    print(myTree)