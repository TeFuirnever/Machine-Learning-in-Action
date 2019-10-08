#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:31
# @Author  : GXl
# @File    : 8.2.2.py
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

# 计算回归系数w
def standRegres(xArr,yArr):
    """
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T * xMat                   #根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 绘制回归曲线和数据点
def plotRegression():
    xArr, yArr = loadDataSet('ex0.txt')   #加载数据集
    ws = standRegres(xArr, yArr)          #计算回归系数
    xMat = np.mat(xArr)                   #创建xMat矩阵
    yMat = np.mat(yArr)                   #创建yMat矩阵
    xCopy = xMat.copy()                   #深拷贝xMat矩阵
    xCopy.sort(0)                         #排序
    yHat = xCopy * ws                     #计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)             #添加subplot
    ax.plot(xCopy[:, 1], yHat, c = 'red') #绘制回归曲线
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5) #绘制样本点
    plt.title('DataSet')                  #绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    plotRegression()
