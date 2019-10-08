#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:40
# @Author  : GXl
# @File    : 5.4.2.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import numpy as np

'''
Parameters:
    无
Returns:
    dataMat - 数据列表
    labelMat - 标签列表
'''
# 函数说明:加载数据
def loadDataSet():
    dataMat = []                                                    #创建数据列表
    labelMat = []                                                   #创建标签列表
    fr = open('testSet.txt')                                        #打开文件
    for line in fr.readlines():                                     #逐行读取
        lineArr = line.strip().split()                              #去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #添加数据
        labelMat.append(int(lineArr[2]))                            #添加标签
    fr.close()                                                      #关闭文件
    return dataMat, labelMat                                        #返回

'''
Parameters:
    inX - 数据
Returns:
    sigmoid函数
'''
# 函数说明:sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

'''
Parameters:
    dataMatIn - 数据集
    classLabels - 数据标签
Returns:
'''
# 函数说明:梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                  #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                      #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                     #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001                                                   #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                 #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                           #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                           #将矩阵转换为数组，返回权重数组


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    print(gradAscent(dataMat, labelMat))