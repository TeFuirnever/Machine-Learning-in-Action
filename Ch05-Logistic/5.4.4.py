#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 14:42
# @Author  : GXl
# @File    : 5.4.4.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

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
    weights - 权重参数数组
Returns:
    无
'''
# 函数说明:绘制数据集
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                   #加载数据集
    dataArr = np.array(dataMat)                                         #转换成numpy的array数组
    n = np.shape(dataMat)[0]                                            #数据个数
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = []                                            #负样本
    for i in range(n):                                                  #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                                #绘制title
    plt.xlabel('X1'); plt.ylabel('X2')                                  #绘制label
    plt.show()

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
    m,n = np.shape(dataMatrix)                                       #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                             #参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                 #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))        #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))          #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                       #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]#更新回归系数
            del(dataIndex[randIndex])                                #删除已经使用的样本
    return weights                                                   #返回


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)