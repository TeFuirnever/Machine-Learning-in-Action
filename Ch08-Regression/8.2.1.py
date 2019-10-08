#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:30
# @Author  : GXl
# @File    : 8.2.1.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import matplotlib.pyplot as plt
import numpy as np

# 加载数据
def loadDataSet(fileName):
    """
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

# 绘制数据集
def plotDataSet():
    xArr, yArr = loadDataSet('ex0.txt')                     #加载数据集
    n = len(xArr)                                           #数据个数
    xcord = []; ycord = []                                  #样本点
    for i in range(n):
        xcord.append(xArr[i][1]); ycord.append(yArr[i])     #样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)                               #添加subplot
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5) #绘制样本点
    plt.title('DataSet')                                    #绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    plotDataSet()
