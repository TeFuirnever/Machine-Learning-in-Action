#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:36
# @Author  : GXl
# @File    : 9.3.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import numpy as np

# 函数说明:根据给定特征和特征值，通过数组过滤的方式切分数据集合
"""
Parameters:
	dataSet - 数据集合
	feature - 待切分的特征
	value - 特征的某个值
"""
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1


if __name__ == '__main__':
    testMat = np.mat(np.eye(4))
    print("testMat：", testMat)
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print("mat0：", mat0)
    print("mat1：", mat1)
