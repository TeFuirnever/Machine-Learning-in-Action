#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:44
# @Author  : GXl
# @File    : 5.5.2.py
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
    dataMatrix - 数据数组
    classLabels - 数据标签
    numIter - 迭代次数
Returns:
    weights - 求得的回归系数数组(最优参数)
'''
# 函数说明:改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)                                         #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                               #参数初始化
    #存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                  #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))         #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))           #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                        #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex] #更新回归系数
            del(dataIndex[randIndex])                                 #删除已经使用的样本
    return weights                                                    #返回

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