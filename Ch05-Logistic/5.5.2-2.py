#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:45
# @Author  : GXl
# @File    : 5.5.2-2.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import numpy as np
import random

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
    weights.getA() - 求得的权重数组(最优参数)
'''
# 函数说明:梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)                                 #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()                     #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)                                    #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01                                                   #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                                                #最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                          #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                                          #将矩阵转换为数组，并返回

# 函数说明:使用Python写的Logistic分类器做预测
def colicTest():
    frTrain = open('horseColicTraining.txt')                                          #打开训练集
    frTest = open('horseColicTest.txt')                                               #打开测试集
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)        #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                                 #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)

'''
Parameters:
    inX - 特征向量
    weights - 回归系数
Returns:
    分类结果
'''
# 函数说明:分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


if __name__ == '__main__':
    colicTest()