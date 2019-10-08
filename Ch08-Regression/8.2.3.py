#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 15:31
# @Author  : GXl
# @File    : 8.2.3.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


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
    xTx = xMat.T * xMat                       #根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')        #加载数据集
    ws = standRegres(xArr, yArr)               #计算回归系数
    xMat = np.mat(xArr)                        #创建xMat矩阵
    yMat = np.mat(yArr)                        #创建yMat矩阵
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat))
